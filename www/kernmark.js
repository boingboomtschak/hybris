
class Kernmark {
    constructor() {
        this.adapter = null;
        this.device = null;
    }
    async initializeGPU() {
        console.log("Initializing WebGPU...");
        this.adapter = await navigator.gpu.requestAdapter();
        this.device = await this.adapter.requestDevice();
    }
    createPipeline(wgslCode, entryPoint = 'main') {
        if (this.adapter === null || this.device === null) return;
        return this.device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: this.device.createShaderModule({
                    code: wgslCode,
                }),
                entryPoint: entryPoint
            }
        });
    }
    async runVectorAdd() {
        if (this.adapter === null || this.device === null) return;
        // Create compute pipeline
        const computePipeline = this.createPipeline();
        // Setup device buffers
        const N = 10^5;
        const bufferSize = N * Float32Array.BYTES_PER_ELEMENT;
        const bufferA = device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        const bufferB = device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        const bufferC = device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });

        // Create data and copy to buffers
        
        
        const kernel = `
            @binding(0) @group(0) var<storage, read> a : array<u32>;
            @binding(1) @group(0) var<storage, read> b : array<u32>;
            @binding(2) @group(0) var<storage, read_write> c : array<u32>;
            @compute
            fn main(@builtin(global_invocation_id) giid : vec3<u32>) {
                c[giid.x] = a[giid.x] + b[giid.x];
            }
        `;
        // need to create buffers, set up binding group, dispatch kernel, retrieve and check results
    }
}
const kernmark = new Kernmark();

document.addEventListener("DOMContentLoaded", kernmark.initializeGPU);
