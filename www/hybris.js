
class Hybris {
    constructor() {
        this.adapter = null;
        this.device = null;
        this.data = { bfs: {} };
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
    async runVectorAdd(N=10**6, MAX_VAL=1024) {
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
        const t_init = window.performance.now();
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
        const a = new Uint32Array(N);
        const b = new Uint32Array(N);
        for (let i = 0; i < N; i++) {
            a[i] = Math.floor(Math.random() * MAX_VAL);
            b[i] = MAX_VAL - a[i];
        }
        this.device.queue.writeBuffer(bufferA, 0, a);
        this.device.queue.writeBuffer(bufferB, 0, b);

        // Create temp buffer to map results
        const bufferMap = this.device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });

        const t_bufcopy = window.performance.now();
        console.log(`Created input arrays and copied to buffers in ${this.getTimeDiff(t_init, t_bufcopy)}`);

        // Create compute pipeline
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
        const bindGroup = this.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: bufferA, offset: 0, size: bufferSize } },
                { binding: 1, resource: { buffer: bufferB, offset: 0, size: bufferSize } },
                { binding: 2, resource: { buffer: bufferC, offset: 0, size: bufferSize } }
            ]
        });

        const t_kernel_setup = window.performance.now();
        console.log(`Set up pipeline and kernel in ${this.getTimeDiff(t_bufcopy, t_kernel_setup)}`);

        // Dispatch kernel and copy bufferC to bufferMap
        const commandEncoder = this.device.createCommandEncoder();
        const computePassEncoder = commandEncoder.beginComputePass();
        computePassEncoder.setPipeline(pipeline);
        computePassEncoder.setBindGroup(0, bindGroup);
        computePassEncoder.dispatchWorkgroups(numWorkgroups);
        computePassEncoder.end();
        commandEncoder.copyBufferToBuffer(bufferC, 0, bufferMap, 0, bufferSize);
        this.device.queue.submit([commandEncoder.finish()]);
        const t_queue_submit = window.performance.now();
        console.log(`Dispatched kernel and copied result buffer in ${this.getTimeDiff(t_kernel_setup, t_queue_submit)}`);

        // Map buffer C and check results
        await bufferMap.mapAsync(GPUMapMode.READ, 0, bufferSize);
        const mapped = new Uint32Array(bufferMap.getMappedRange());
        let incorrect = false;
        for (let i = 0; i < N; i++)
            if (mapped[i] != MAX_VAL) 
                incorrect = true;
        bufferMap.unmap();
        let t_check = window.performance.now();
        console.log(`Checked results in ${this.getTimeDiff(t_queue_submit, t_check)}`);

        if (!incorrect)
            console.log(`${N} results checked, correct!`);
        else
            console.log(`${N} results checked, incorrect!`);
    }
    async runBFS(data='data/bfs/graph1MW_6.txt') {
        if (this.adapter === null || this.device === null) {
            console.log("Hybris not initialized!");
            return; 
        }

        const MAX_THREADS_PER_BLOCK = this.device.limits.maxComputeWorkgroupSizeX;
        let no_of_nodes = 0;
        let edge_list_size = 0;
        let source = 0;

        const t_init = window.performance.now();

        let text;
        if (data in this.data.bfs) {
            text = this.data.bfs[data];
        } else {
            const response = await fetch(data);
            text = await response.text();
            this.data.bfs[data] = text;
            const t_fetch = window.performance.now();
            console.log(`Retrieved ${data} in ${this.getTimeDiff(t_init, t_fetch)}`);
        }
        const t_retrieve = window.performance.now();
        text = text.split(/[\r\n ]+/);
        text = text.reverse();
        const t_parsed = window.performance.now();
        console.log(`Parsed ${data} in ${this.getTimeDiff(t_retrieve, t_parsed)}`);

        // Reading graph nodes and setting up graph mask/visited arrays
        no_of_nodes = parseInt(text.pop());
        const h_graph_nodes_starting = new Uint32Array(no_of_nodes);
        const h_graph_nodes_no_of_edges = new Uint32Array(no_of_nodes);
        const h_graph_mask = new Uint8Array(no_of_nodes);
        const h_updating_graph_mask = new Uint8Array(no_of_nodes);
        const h_graph_visited = new Uint8Array(no_of_nodes);
        for (let i = 0; i < no_of_nodes; i++) {
            h_graph_nodes_starting[i] = text.pop();
            h_graph_nodes_no_of_edges[i] = text.pop();
            h_graph_mask[i] = 0;
            h_updating_graph_mask[i] = 0;
            h_graph_visited[i] = 0;
        }
        const t_nodes = window.performance.now();
        console.log(`Processed ${no_of_nodes} nodes in ${this.getTimeDiff(t_parsed, t_nodes)}`);
        
        // Reading graph source
        let source_line = text.pop();
        // Rodinia bfs does not even read source node from file! Sets to 0 in all cases
        //source = (isNaN(source_line)) ? source : parseInt(source_line);
        source = 0;
        
        // Reading graph edge list size
        let edge_list_size_line = text.pop();
        edge_list_size = (isNaN(edge_list_size_line)) ? edge_list_size : parseInt(edge_list_size_line);
        
        // Reading graph edges
        let h_graph_edges = new Uint32Array(edge_list_size);
        for (let i = 0; i < edge_list_size; i++) {
            h_graph_edges[i] = text.pop();
            text.pop();
        }
        const t_edges = window.performance.now();
        console.log(`Processed ${edge_list_size} edges in ${this.getTimeDiff(t_nodes, t_edges)}`);

        let num_of_blocks = 1;
        let num_of_threads_per_block = no_of_nodes;
        if (no_of_nodes > MAX_THREADS_PER_BLOCK) {
            num_of_blocks = Math.ceil(no_of_nodes/MAX_THREADS_PER_BLOCK);
            num_of_threads_per_block = MAX_THREADS_PER_BLOCK;
        }

        h_graph_mask[source] = 1;
        h_graph_visited[source] = 1;

        // Creating device buffers for graph
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
            size: no_of_nodes*Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(d_graph_mask, 0, h_graph_mask);
        const d_updating_graph_mask = this.device.createBuffer({
            size: no_of_nodes*Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(d_updating_graph_mask, 0, h_updating_graph_mask);
        const d_graph_visited = this.device.createBuffer({
            size: no_of_nodes*Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(d_graph_visited, 0, h_graph_visited);

        // Create host and device buffer for result
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

        // Create buffer(s) for stop/d_stop
        const h_stop = new Uint32Array(1);
        h_stop[0] = 0;
        const d_stop = this.device.createBuffer({
            size: Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });
        this.device.queue.writeBuffer(d_stop, 0, h_stop);
        const d_stop_map = this.device.createBuffer({
            size: Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });

        const t_h2d = window.performance.now();
        console.log(`Copied graph data from host to device in ${this.getTimeDiff(t_edges, t_h2d)}`);

        // Create pipeline/binding group layout/kernel

        const kernel = `
            @binding(0) @group(0) var<storage, read> g_graph_nodes_starting : array<u32>;
            @binding(1) @group(0) var<storage, read> g_graph_nodes_no_of_edges : array<u32>;
            @binding(2) @group(0) var<storage, read> g_graph_edges : array<u32>;
            @binding(3) @group(0) var<storage, read_write> g_graph_mask : array<u32>;
            @binding(4) @group(0) var<storage, read_write> g_updating_graph_mask : array<u32>;
            @binding(5) @group(0) var<storage, read> g_graph_visited : array<u32>;
            @binding(6) @group(0) var<storage, read_write> g_cost : array<u32>;
            @compute @workgroup_size(${num_of_threads_per_block})
            fn main(@builtin(workgroup_id) wg_id : vec3<u32>, @builtin(local_invocation_id) liid : vec3<u32>) {
                var tid : u32 = wg_id.x * ${MAX_THREADS_PER_BLOCK} + liid.x;
                if (tid < ${no_of_nodes} && g_graph_mask[tid] == 1) {
                    g_graph_mask[tid] = 0;
                    for (var i : u32 = g_graph_nodes_starting[tid]; i < (g_graph_nodes_no_of_edges[tid] + g_graph_nodes_starting[tid]); i++) {
                        var id : u32 = g_graph_edges[i];
                        if (g_graph_visited[id] == 0) {
                            g_cost[id] = g_cost[tid] + 1;
                            g_updating_graph_mask[id] = 1;
                        }
                    }
                }
            }
        `;

        const kernel2 = `
            @binding(0) @group(0) var<storage, read_write> g_updating_graph_mask : array<u32>;
            @binding(1) @group(0) var<storage, read_write> g_graph_mask : array<u32>;
            @binding(2) @group(0) var<storage, read_write> g_graph_visited : array<u32>;
            @binding(3) @group(0) var<storage, read_write> g_stop : array<u32>;
            @compute @workgroup_size(${num_of_threads_per_block})
            fn main(@builtin(workgroup_id) wg_id : vec3<u32>, @builtin(local_invocation_id) liid : vec3<u32>) {
                var tid : u32 = wg_id.x * ${MAX_THREADS_PER_BLOCK} + liid.x;
                if (tid < ${no_of_nodes} && g_updating_graph_mask[tid] == 1) {
                    g_graph_mask[tid] = 1;
                    g_graph_visited[tid] = 1;
                    g_stop[0] = 1;
                    g_updating_graph_mask[tid] = 0;
                }
            }
        `;
        const kernelPipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: this.device.createShaderModule({ code: kernel }),
                entryPoint: 'main'
            }
        });
        const kernel2Pipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: this.device.createShaderModule({ code: kernel2 }),
                entryPoint: 'main'
            }
        });

        const kernelBindGroup = this.device.createBindGroup({
            layout: kernelPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: d_graph_nodes_starting, offset: 0, size: no_of_nodes*Uint32Array.BYTES_PER_ELEMENT } },
                { binding: 1, resource: { buffer: d_graph_nodes_no_of_edges, offset: 0, size: no_of_nodes*Uint32Array.BYTES_PER_ELEMENT } },
                { binding: 2, resource: { buffer: d_graph_edges, offset: 0, size: edge_list_size*Uint32Array.BYTES_PER_ELEMENT } },
                { binding: 3, resource: { buffer: d_graph_mask, offset: 0, size: no_of_nodes*Uint32Array.BYTES_PER_ELEMENT } },
                { binding: 4, resource: { buffer: d_updating_graph_mask, offset: 0, size: no_of_nodes*Uint32Array.BYTES_PER_ELEMENT } },
                { binding: 5, resource: { buffer: d_graph_visited, offset: 0, size: no_of_nodes*Uint32Array.BYTES_PER_ELEMENT } },
                { binding: 6, resource: { buffer: d_cost, offset: 0, size: no_of_nodes*Uint32Array.BYTES_PER_ELEMENT } }
            ]
        });
        const kernel2BindGroup = this.device.createBindGroup({
            layout: kernel2Pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: d_updating_graph_mask, offset: 0, size: no_of_nodes*Uint32Array.BYTES_PER_ELEMENT } },
                { binding: 1, resource: { buffer: d_graph_mask, offset: 0, size: no_of_nodes*Uint32Array.BYTES_PER_ELEMENT } },
                { binding: 2, resource: { buffer: d_graph_visited, offset: 0, size: no_of_nodes*Uint32Array.BYTES_PER_ELEMENT } },
                { binding: 3, resource: { buffer: d_stop, offset: 0, size: Uint32Array.BYTES_PER_ELEMENT } }
            ]
        });

        console.log(`Dispatching ${num_of_blocks} workgroups of size ${num_of_threads_per_block}...`);

        let k = 0;
        const t_prekernel = window.performance.now();
        do {
            stop = false;
            // copy stop to d_stop
            h_stop[0] = (stop) ? 1 : 0;
            this.device.queue.writeBuffer(d_stop, 0, h_stop);

            const commandEncoder = this.device.createCommandEncoder();
            // dispatch kernels
            const kernelPass = commandEncoder.beginComputePass();
            kernelPass.setPipeline(kernelPipeline);
            kernelPass.setBindGroup(0, kernelBindGroup);
            kernelPass.dispatchWorkgroups(num_of_blocks);
            kernelPass.end()
            const kernel2Pass = commandEncoder.beginComputePass();
            kernel2Pass.setPipeline(kernel2Pipeline);
            kernel2Pass.setBindGroup(0, kernel2BindGroup);
            kernel2Pass.dispatchWorkgroups(num_of_blocks);
            kernel2Pass.end();
            // copy d_stop to d_stop_map
            commandEncoder.copyBufferToBuffer(d_stop, 0, d_stop_map, 0, Uint32Array.BYTES_PER_ELEMENT);
            this.device.queue.submit([commandEncoder.finish()]);

            // map d_stop_map, read val
            await d_stop_map.mapAsync(GPUMapMode.READ, 0, Uint32Array.BYTES_PER_ELEMENT);
            const mapped = new Uint32Array(d_stop_map.getMappedRange());
            stop = mapped[0] == 1 ? true : false;
            d_stop_map.unmap();

            k++;
        } while (stop);
        const t_postkernel = window.performance.now();
        console.log(`Dispatched kernels ${k} times in ${this.getTimeDiff(t_prekernel, t_postkernel)}`);
        
        // Copy cost back, read result file, check results
        const resultCommandEncoder = this.device.createCommandEncoder();
        resultCommandEncoder.copyBufferToBuffer(d_cost, 0, d_cost_map, 0, no_of_nodes*Uint32Array.BYTES_PER_ELEMENT);
        this.device.queue.submit([resultCommandEncoder.finish()]);

        let results = await fetch(`data/bfs/result-${data.split('/').at(-1)}`);
        results = await results.text();
        let resultIter = results.matchAll(/[0-9]+\) cost:([0-9]+)\r?\n?/g);
        
        let correct = true;
        await d_cost_map.mapAsync(GPUMapMode.READ, 0, no_of_nodes*Uint32Array.BYTES_PER_ELEMENT);
        const mapped = new Uint32Array(d_cost_map.getMappedRange());
        for (let i = 0; i < no_of_nodes; i++) {
            let r = resultIter.next().value[1];
            if (mapped[i] != r) {
                correct = false;
                break;
            }
        }
        d_cost_map.unmap(); 
        const t_check = window.performance.now();
        if (correct)
            console.log(`Checked results in ${this.getTimeDiff(t_postkernel, t_check)}, all correct!`);
        else
            console.log('Results incorrect!');
        console.log(`Total execution time took ${this.getTimeDiff(t_init, t_check)}`);
    }
    // Debug, from https://stackoverflow.com/questions/3665115/how-to-create-a-file-in-memory-for-user-to-download-but-not-through-server
    download(filename, text) {
        let element = document.createElement('a');
        element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
        element.setAttribute('download', filename);
        element.style.display = 'none';
        document.body.appendChild(element);
        element.click();
        document.body.removeChild(element);
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
    getTimeDiff(t_start, t_end) {
        const diff = t_end - t_start;
        return (diff > 1000) ? `${(diff / 1000).toFixed(3)} s` : `${diff.toFixed(1)} ms`;
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


