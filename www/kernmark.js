
class Kernmark {
    constructor() {
        this.adapter = null;
        this.device = null;
    }
    async initializeGPU() {
        console.log("Initializing Kernmark...");
        if (!navigator.gpu) {
            alert("ERROR: WebGPU not enabled!");
            return;
        }
        this.adapter = await navigator.gpu.requestAdapter();
        this.device = await this.adapter.requestDevice();
        if (this.adapter && this.device) console.log("Kernmark initialized!");
        else console.log("Error initializing Kernmark!");
    }
    async runVectorAdd(N=10**5, MAX_VAL=1024) {
        if (this.adapter === null || this.device === null) {
            console.log("Kernmark not initialized!");
            return; 
        }

        // Use maximum workgroup size available, scale number of workgroups as needed
        const numWorkgroups = Math.ceil(N / this.device.limits.maxComputeWorkgroupSizeX);
        const workgroupSize = this.device.limits.maxComputeWorkgroupSizeX;
        
        if (numWorkgroups > this.device.limits.maxComputeWorkgroupsPerDimension) {
            console.log("N too large, max workgroups exceeded!");
            return;
        }
        if ((N * 4) > this.device.limits.maxStorageBufferBindingSize) {
            console.log("N too large, storage buffer binding size exceeded!")
            return;
        }

        // Setup device buffers
        console.log("Creating device buffers...");
        const bufferSize = N * Uint32Array.BYTES_PER_ELEMENT;
        const bufferA = this.device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        const bufferB = this.device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        const bufferC = this.device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });

        // Create data and copy to buffers
        console.log("Creating local arrays and copying to buffers...");
        const a = new Uint32Array(N);
        const b = new Uint32Array(N);
        for (let i = 0; i < N; i++) {
            a[i] = Math.floor(Math.random() * MAX_VAL);
            b[i] = MAX_VAL - a[i];
        }
        this.device.queue.writeBuffer(bufferA, 0, a);
        this.device.queue.writeBuffer(bufferB, 0, b);

        // Create compute pipeline
        console.log("Creating compute pipeline...");
        const kernel = `
            @binding(0) @group(0) var<storage, read> a : array<u32>;
            @binding(1) @group(0) var<storage, read> b : array<u32>;
            @binding(2) @group(0) var<storage, read_write> c : array<u32>;
            @compute @workgroup_size(${workgroupSize})
            fn main(@builtin(global_invocation_id) giid : vec3<u32>) {
                c[giid.x] = a[giid.x] + b[giid.x];
            }
        `;
        const pipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: this.device.createShaderModule({ code: kernel }),
                entryPoint: 'main'
            }
        });

        // Create binding group
        console.log("Creating binding group...");
        const bindGroup = this.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: bufferA,
                        offset: 0,
                        size: bufferSize
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: bufferB,
                        offset: 0,
                        size: bufferSize
                    }
                },
                {
                    binding: 2,
                    resource: {
                        buffer: bufferC,
                        offset: 0,
                        size: bufferSize
                    }
                }
            ]
        });

        // Create temp buffer to map and read results
        const bufferMap = this.device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });

        // Dispatch kernel and copy bufferC to bufferMap
        console.log("Dispatching kernel and copying result buffer...");
        const commandEncoder = this.device.createCommandEncoder();
        const computePassEncoder = commandEncoder.beginComputePass();
        computePassEncoder.setPipeline(pipeline);
        computePassEncoder.setBindGroup(0, bindGroup);
        computePassEncoder.dispatchWorkgroups(numWorkgroups);
        computePassEncoder.end();
        commandEncoder.copyBufferToBuffer(bufferC, 0, bufferMap, 0, bufferSize);
        this.device.queue.submit([commandEncoder.finish()]);

        // Map buffer C and check results
        console.log("Mapping buffer and checking results...");
        await bufferMap.mapAsync(GPUMapMode.READ, 0, bufferSize);
        const mapped = new Uint32Array(bufferMap.getMappedRange());
        let incorrect = false;
        for (let i = 0; i < N; i++)
            if (mapped[i] != MAX_VAL) 
                incorrect = true;
        bufferMap.unmap();

        if (!incorrect)
            console.log(`${N} results checked, correct!`);
        else
            console.log(`${N} results checked, incorrect!`);
    }
}
const kernmark = new Kernmark();

document.addEventListener("DOMContentLoaded", () => {
    kernmark.initializeGPU()
});
