@import Foundation;
@import CoreGraphics.CGDirectDisplayMetal;
@import Metal;

struct SubP {
    uint mid, diag1, diag2;
    uint ss;
};

struct ProgressStore {
    uint i;
    uint mid;
    uint diag1, diag1r;
    uint diag2, diag2r;
    uint can;
    uint s0[12];
};

id<MTLComputePipelineState> Nqueen_kern;
id<MTLCommandQueue> Command_queue;

id<MTLBuffer> gpu_lv;
id<MTLBuffer> gpu_canplace;
id<MTLBuffer> gpu_works;
id<MTLBuffer> gpu_workCount;
id<MTLBuffer> gpu_flag;
id<MTLBuffer> gpu_result;
id<MTLBuffer> gpu_progress;
struct ProgressStore *gpu_progress_backup;

id<MTLCommandBuffer> nqueen_send_gpu(int cut, int nblocks, int nworks, struct SubP *works, bool recompute) {
    int flag = 0;
    struct SubP *works_data = gpu_works.contents;
    if (!recompute) {
        memcpy(works_data, works, sizeof(struct SubP) * nworks);
        memcpy(gpu_progress_backup, gpu_progress.contents, sizeof(struct ProgressStore) * nblocks*256);
    } else {
        memcpy(gpu_progress.contents, gpu_progress_backup, sizeof(struct ProgressStore) * nblocks*256);
    }
    //cudaMemcpy(gpu_flag, &flag, sizeof(int), cudaMemcpyHostToDevice);
    //cudaMemcpy(gpu_works, works, nworks * sizeof(SubP), cudaMemcpyHostToDevice);
    
    *(int*) gpu_lv.contents = cut;
    *(int*) gpu_workCount.contents = nworks;
    *(int*) gpu_flag.contents = flag;
    
    id<MTLCommandBuffer> commandBuffer = [Command_queue commandBuffer];
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setComputePipelineState:Nqueen_kern];
    [computeEncoder setBuffer:gpu_lv offset:0 atIndex:0];
    [computeEncoder setBuffer:gpu_canplace offset:0 atIndex:1];
    [computeEncoder setBuffer:gpu_works offset:0 atIndex:2];
    [computeEncoder setBuffer:gpu_workCount offset:0 atIndex:3];
    [computeEncoder setBuffer:gpu_flag offset:0 atIndex:4];
    [computeEncoder setBuffer:gpu_result offset:0 atIndex:5];
    [computeEncoder setBuffer:gpu_progress offset:0 atIndex:6];
    [computeEncoder setThreadgroupMemoryLength:sizeof(uint)*12*256 atIndex:0];

    MTLSize gridSize = MTLSizeMake(nblocks * 256, 1, 1);
    MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);

    [computeEncoder dispatchThreads:gridSize
        threadsPerThreadgroup:threadgroupSize];
    
    [computeEncoder endEncoding];
    [commandBuffer commit];
    return commandBuffer;
}

int errCount = 0;
long long nqueen_wait_compute(int nblocks, id<MTLCommandBuffer> commandBuffer, int *finished) {
    uint *result = gpu_result.contents;
    long long sum = 0;

    [commandBuffer waitUntilCompleted];
    if (errCount > 10) return -2;
    if ([commandBuffer error]) {
        NSLog(@"error %@", [commandBuffer error]);
        NSLog(@"%ld", (long)[[commandBuffer error] code]);
        errCount++;
        return -1;
    }
    //cudaMemcpy(result, gpu_result, nblocks*256 * sizeof(long long), cudaMemcpyDeviceToHost);
    //printf("%f %f\n", [commandBuffer GPUStartTime], [commandBuffer GPUEndTime]);
    for (int i = 0; i < nblocks*256; i++) {
        sum += result[i];
    }
    *finished = *(int *)gpu_flag.contents;
    return sum;
}

long long nqueen_gen(int lv, int cut, uint mid, uint diag1, uint diag2,
        uint *canplace, int nblocks, int nworks) {
    struct SubP *works = malloc(sizeof(struct SubP) * nworks*2);
    int worksCount = 0;
    uint s0[32], s1[32], s2[32], s3[32];
    if (lv < cut) return 0;
    int i = lv-1;
    uint choice = canplace[lv-1] & ~(mid | diag1 | diag2);
    s0[i] = choice;
    s1[i] = mid;
    s2[i] = diag1;
    s3[i] = diag2;
    long long sum = 0;
    id<MTLCommandBuffer> lastWork = nil;
    int consumed = 0;
    while (i < lv) {
        choice = s0[i];
        mid = s1[i];
        diag1 = s2[i];
        diag2 = s3[i];
        uint bit = choice & -choice;
        s0[i] = choice - bit;
        if (!choice) i++;
        else if (i == cut-1) {
            struct SubP p = {mid, diag1, diag2};
            works[worksCount++] = p;
            if (worksCount == nworks || worksCount == nworks*2) {
                if (lastWork) {
                    do {
                        long long result = nqueen_wait_compute(nblocks, lastWork, &consumed);
                        while (result == -1) {
                            lastWork = nqueen_send_gpu(cut, nblocks, nworks, works, true);
                            result = nqueen_wait_compute(nblocks, lastWork, &consumed);
                        }
                        consumed = MIN(consumed, nworks);
                        memmove(works, works + consumed, sizeof(struct SubP) * (worksCount - consumed));
                        worksCount -= consumed;
                        sum += result;
                    } while (consumed == 0) ;
                }
                lastWork = nqueen_send_gpu(cut, nblocks, nworks, works, false);
            }
            i++;
        }
        else {
            mid = mid + bit;
            diag1 = (diag1 | bit) << 1;
            diag2 = (diag2 | bit) >> 1;
            i -= 1;
            choice = canplace[i] & ~(mid | diag1 | diag2);
            s0[i] = choice;
            s1[i] = mid;
            s2[i] = diag1;
            s3[i] = diag2;
            if (choice == 0) i += 1;
        }
    }
    if (lastWork) {
        long long result = nqueen_wait_compute(nblocks, lastWork, &consumed);
        while (result == -1) {
            lastWork = nqueen_send_gpu(cut, nblocks, nworks, works, true);
            result = nqueen_wait_compute(nblocks, lastWork, &consumed);
        }
        consumed = MIN(consumed, nworks);
        sum += result;
        memmove(works, works + consumed, sizeof(struct SubP) * (worksCount - consumed));
        worksCount -= consumed;
    }
    bool all_finish = false;
    while (worksCount > 0 || !all_finish) {
        int wc = MIN(worksCount, nworks);
        lastWork = nqueen_send_gpu(cut, nblocks, wc, works, false);
        long long result = nqueen_wait_compute(nblocks, lastWork, &consumed);
        while (result == -1) {
            lastWork = nqueen_send_gpu(cut, nblocks, wc, works, true);
            result = nqueen_wait_compute(nblocks, lastWork, &consumed);
        }
        consumed = MIN(consumed, wc);
        sum += result;
        if (worksCount > consumed) {
            memmove(works, works + consumed, sizeof(struct SubP) * (worksCount - consumed));
        }
        worksCount -= consumed;
        all_finish = true;
        struct ProgressStore *ps = gpu_progress.contents;
        for (int i = 0; i < nblocks*256; i++) {
            if (ps[i].i != 87) all_finish = false;
        }
    }
    free(works);
    return sum;
}

long long nqueen_metal(int n, unsigned canplace[], int nblocks, int nworks) {
    // create subproblems
    int cut = n - 5;
    if (cut < 2) cut = 2;
    if (cut > 13) cut = 13;
    uint *canplace_data = gpu_canplace.contents;
    for (int i = 0; i < 32; i++) {
        canplace_data[i] = canplace[i];
    }
    struct ProgressStore *ps = gpu_progress.contents;
    for (int i = 0; i < nblocks*256; i++) {
        ps[i].i = 87;
    }
    //cudaMemcpy(gpu_canplace, canplace, sizeof(uint) * 32, cudaMemcpyHostToDevice);
    return nqueen_gen(n, cut, 0, 0, 0, canplace, nblocks, nworks);
}

int main(int argc, char *argv[]) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        NSLog(@"No Device!!!");
        return 1;
    }
    //NSLog(@"Yes Device :-)");

    NSError *libraryError = NULL;
    NSString *libraryFile = @"kernel.metallib";
    id <MTLLibrary> myLibrary = [device newLibraryWithFile:libraryFile error:&libraryError];
    if (!myLibrary) {
        NSLog(@"Library error: %@", libraryError.localizedDescription);
        return 2;
    }
    //NSLog(@"Yes Library :-)");

    id<MTLFunction> nqueen_kern_func = [myLibrary newFunctionWithName:@"nqueen_kern"];
    if (nqueen_kern_func == nil)
    {
        NSLog(@"Failed to find the adder function.");
        return 3;
    }
    //NSLog(@"Yes Function :-)");

    NSError *error = NULL;
    Nqueen_kern = [device newComputePipelineStateWithFunction: nqueen_kern_func error:&error];

    Command_queue = [device newCommandQueue];
    //NSLog(@"Created Command queue");

    int n;
    int T = 0;
    char buf[100];
    unsigned canplace[32] = {0};
    FILE *filein = stdin;
    for (int i = 1; i < argc; i++) {
        if (i+1<argc && filein == stdin && strcmp(argv[i], "-i") == 0) {
            filein = fopen(argv[i+1], "r");
            if (filein == NULL) {
                fprintf(stderr, "cannot open file\n");
                return 1;
            }
        }
    }

    // Metal has no API for retrieving hardware unit count
    int multiProcessorCount = 8;
    int nblocks = multiProcessorCount * 5;
    int workCount = nblocks * 256 * 7;

    gpu_lv = [device newBufferWithLength:sizeof(int) options:MTLResourceStorageModeShared];
    gpu_canplace = [device newBufferWithLength:sizeof(uint)*32 options:MTLResourceStorageModeShared];
    gpu_works = [device newBufferWithLength:sizeof(struct SubP)*workCount options:MTLResourceStorageModeShared];
    gpu_workCount = [device newBufferWithLength:sizeof(int) options:MTLResourceStorageModeShared];
    gpu_flag = [device newBufferWithLength:sizeof(uint) options:MTLResourceStorageModeShared];
    gpu_result = [device newBufferWithLength:sizeof(uint)*nblocks*256 options:MTLResourceStorageModeShared];
    gpu_progress = [device newBufferWithLength:sizeof(struct ProgressStore)*nblocks*256 options:MTLResourceStorageModeShared];
    gpu_progress_backup = malloc(sizeof(struct ProgressStore)*nblocks*256);
    //NSLog(@"Created Buffer");

    while (fscanf(filein, "%d", &n) == 1) {
        fgets(buf, 100, filein);
        T += 1;
        if (n < 1 || n >= 32) return 0;
        for (int i = 0; i < n; i++) {
            fgets(buf, 100, filein);
            canplace[n-1-i] = (1u<<n)-1;
            for (int j = 0; j < n; j++) {
                if (buf[j] == '*') canplace[n-1-i] -= 1u<<j;
            }
        }

        long long ans = 0;
        ans = nqueen_metal(n, canplace, nblocks, workCount);
        printf("Case #%d: %lld\n", T, ans);
    }
    return 0;
}
