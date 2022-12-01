
class Hybris {
    constructor() {
        this.adapter = null;
        this.device = null;
        this.bfsGraph = null;
    }
    async initializeGPU() {
        console.log("Initializing Hybris...");
        if (!("gpu" in navigator)) {
            alert("ERROR: WebGPU not enabled!");
            return;
        }
        this.adapter = await navigator.gpu.requestAdapter({
            powerPreference : "high-performance"
        });
        this.device = await this.adapter.requestDevice();
        const adapterInfo = await this.adapter.requestAdapterInfo();
        
        document.getElementById("adapter-device").innerHTML = (adapterInfo.device) ? adapterInfo.device : "Unknown";
        document.getElementById("adapter-desc").innerHTML =  (adapterInfo.description) ? adapterInfo.description : "Unknown";
        document.getElementById("adapter-arch").innerHTML = (adapterInfo.architecture) ? adapterInfo.architecture : "Unknown";
        document.getElementById("adapter-vendor").innerHTML = (adapterInfo.vendor) ? adapterInfo.vendor : "Unknown";
        document.getElementById("status").innerHTML = (this.adapter && this.device) ? "Initialized" : "Error initializing!";
        console.log("Initialized!");
        console.log("----------------");
    }
    async runVectorAdd(N=10**5, MAX_VAL=1024) {
        if (this.adapter === null || this.device === null) {
            console.log("Hybris not initialized!");
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
    async runBFS() {
        if (this.adapter === null || this.device === null) {
            console.log("Hybris not initialized!");
            return; 
        }

        const MAX_THREADS_PER_BLOCK = 512;
        let no_of_nodes = 0;
        let edge_list_size = 0;
        let source = 0;

        console.log("Retrieving and processing data/graph65536...");   
        const response = await fetch('/data/graph65536.txt');
        let text = await response.text();
        text = text.split(/\r?\n/);

        // Reading graph nodes and setting up graph mask/visited arrays
        no_of_nodes = parseInt(text.shift());
        const h_graph_nodes_starting = new Uint32Array(no_of_nodes);
        const h_graph_nodes_no_of_edges = new Uint32Array(no_of_nodes);
        const h_graph_mask = new Uint8Array(no_of_nodes);
        const h_updating_graph_mask = new Uint8Array(no_of_nodes);
        const h_graph_visited = new Uint8Array(no_of_nodes);
        for (let i = 0; i < no_of_nodes; i++) {
            let line = text.shift().split(" ");
            h_graph_nodes_starting[i] = line[0];
            h_graph_nodes_no_of_edges[i] = line[1];
            h_graph_mask[i] = 0;
            h_updating_graph_mask[i] = 0;
            h_graph_visited[i] = 0;
        }
        
        // Reading graph source
        text.shift();
        let source_line = text.shift();
        source = (isNaN(source_line)) ? source : parseInt(source_line);
        
        // Reading graph edge list size
        text.shift();
        let edge_list_size_line = text.shift();
        edge_list_size = (isNaN(edge_list_size_line)) ? edge_list_size : parseInt(edge_list_size_line);
        
        // Reading graph edges
        let h_graph_edges = new Uint32Array(edge_list_size);
        for (let i = 0; i < edge_list_size; i++) {
            let line = text.shift().split(" ");
            h_graph_edges[i] = line[0];
        }
        console.log("Processed data/graph65536.");

        let num_of_blocks = 1;
        let num_of_threads_per_block = no_of_nodes;
        if (no_of_nodes > MAX_THREADS_PER_BLOCK) {
            num_of_blocks = Math.ceil(no_of_nodes/MAX_THREADS_PER_BLOCK);
            num_of_threads_per_block = MAX_THREADS_PER_BLOCK;
        }

        h_graph_mask[source] = 1;
        h_graph_visited[source] = 1;

        // Creating device buffers
        console.log("Creating buffers for graph and copying to device memory...");
        const d_graph_nodes_starting = this.device.createBuffer({
            size: no_of_nodes*Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(d_graph_nodes_starting, 0, h_graph_nodes_starting);
        const d_graph_nodes_no_of_edges = this.device.createBuffer({
            size: no_of_nodes*Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(d_graph_nodes_no_of_edges, 0, h_graph_nodes_no_of_edges);
        const d_graph_edges = this.device.createBuffer({
            size: edge_list_size*Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(d_graph_edges, 0, h_graph_edges);
        const d_graph_mask = this.device.createBuffer({
            size: no_of_nodes*Uint8Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffeR(d_graph_mask, 0, h_graph_mask);
        const d_updating_graph_mask = this.device.createBuffer({
            size: no_of_nodes*Uint8Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(d_updating_graph_mask, 0, h_updating_graph_mask);
        const d_graph_visited = this.device.createBuffer({
            size: no_of_nodes*Uint8Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(d_graph_visited, 0, h_graph_visited);

        // Create host and device buffer for result
        console.log("Creating host and device memory for result...");
        const h_cost = new Uint32Array(no_of_nodes);
        for (let i = 0; i < no_of_nodes; i++) 
            h_cost[i] = -1;
        h_cost[source] = 0;
        const d_cost = this.device.createBuffer({
            size: no_of_nodes*Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });
        this.device.queue.writeBuffer(d_cost, 0, h_cost);
        const d_cost_map = this.device.createBuffer({
            size: no_of_nodes*Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });

        // Set up buffer(s) for stop/d_over 

        // Create pipeline/binding group layout/kernel

        console.log("Running kernels...");
        let stop;
        do {
            stop = false;
            
            const commandEncoder = this.device.createCommandEncoder();
            const computePassEncoder = commandEncoder.beginComputePass();
            // dispatch kernels
            computePassEncoder.end();
            this.device.queue.submit([commandEncoder.finish()]);

        } while (stop);
    }
    async runBenchmark(benchmark) {
        console.log(`Running ${benchmark}...`);
        switch(benchmark) {
            case "vector-add": await this.runVectorAdd(); break;
            case "bfs": await this.runBFS(); break;
            case "gaussian": console.log("Not yet implemented!"); break;
            case "particlefilter": console.log("Not yet implemented!"); break;
            default: console.log("Unknown benchmark!"); break;
        }
        console.log(`Finished ${benchmark}.`);
        console.log("----------------");
    }
}

class Console {
    constructor() {
        this.textarea = null;
    }
    getTextarea(element) {
        if (!element) return;
        this.textarea = element;
    }
    write(content) {
        if (!this.textarea) return;
        if (typeof content === 'string') 
            this.textarea.value += (this.textarea.value == "" ? "" : "\n") + content;
        else
            this.textarea.value += (this.textarea.value == "" ? "" : "\n") + `${JSON.stringify(content)}`;
        
        this.textarea.scrollTop = this.textarea.scrollHeight;
    }
    clear() { 
        if (!this.textarea) return;
        this.textarea.value = ""; 
    }
}

const hybris = new Hybris();
const hConsole = new Console();

document.addEventListener("DOMContentLoaded", () => {
    hConsole.getTextarea(document.getElementById('console'));
    document.getElementById("console-clear").onclick = () => { hConsole.clear() };
    document.getElementById("bm-vector-add").onclick = () => { hybris.runBenchmark("vector-add"); }
    document.querySelectorAll('[id^=bm]').forEach((bm) => {
        bm.onclick = () => { hybris.runBenchmark(bm.id.slice(3)); }
    });
    (()=>{
        const console_log = window.console.log;
        window.console.log = function(...args) {
            console_log(...args);
            args.forEach((a) => hConsole.write(a));
        }
    })();
    hybris.initializeGPU();
});


