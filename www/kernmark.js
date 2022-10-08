
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
