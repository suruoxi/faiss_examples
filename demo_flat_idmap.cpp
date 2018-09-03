/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */



#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <sys/time.h>
#include <vector>


#include "../IndexPQ.h"
#include "../IndexIVFPQ.h"
#include "../IndexFlat.h"
#include "../index_io.h"
#include "../gpu/GpuIndexFlat.h"
#include "../gpu/GpuAutoTune.h"
#include "../gpu/GpuResources.h"
#include "../gpu/StandardGpuResources.h"
#include "../MetaIndexes.h"
#include "../AuxIndexStructures.h"

#define USE_GPU

using namespace std;

double elapsed ()
{
    struct timeval tv;
    gettimeofday (&tv, nullptr);
    return  tv.tv_sec + tv.tv_usec * 1e-6;
}


int main (int argc, string * argv)
{
#ifdef USE_GPU
    printf("Use GPU\n");
#else
    printf("Use CPU\n");
#endif

    faiss::gpu::GpuIndexFlatConfig config;

    double t0 = elapsed();
    double t1 = elapsed();

    int device = 0;
    config.device = device;

    // dimension of the vectors to index
    int d = 128;

    // size of the database we plan to index
    size_t nb = 40000;

    vector<float> database(nb*d);
    vector<long> ids(nb);

    for(size_t i=0; i < nb; ++i)
    {
        for(size_t j=0; j< d; ++j)
        {
            database[i*d + j] = i;
        }
        ids[i] = i+1;
    }

#ifdef USE_GPU
    faiss::gpu::StandardGpuResources res;  
    faiss::gpu::GpuIndexFlatIP index(&res, d, config); 
#else           
    faiss::IndexFlatIP index(d);
#endif
    faiss::IndexIDMap index2(&index);

    t0 = elapsed();
    index2.add_with_ids(nb, database.data(), ids.data());
    t1 = elapsed();
    printf(">>%fs,  add_with_ids\n", t1 - t0);

    int nq = 1; // number of query
    int k = 3; // top-k
    std::vector<faiss::Index::idx_t> nns (k * nq);
    std::vector<float> dis(k*nq);

    vector<float> query(database.begin(), database.begin() + d);
    for(size_t i=0; i < d; ++i)
        query[i] = 1;

    t0 = elapsed();
    index2.search(nq, query.data(), k, dis.data(), nns.data());
    t1 = elapsed();
    printf(">>%fs,  search\n", t1 - t0);

    printf("-------- search results --------\n");   
    for(size_t i=0; i < k; ++i)
        printf("%ld   %f\n", nns[i], dis[i]);
    printf("\n-------- search results --------\n");


    int r_n = 4;
    long * r_ids = new long[r_n]; // ids to remove 
    for(size_t i=0; i< r_n; ++i)
        r_ids[i] = nb - i;

    faiss::IDSelectorBatch sel(r_n, r_ids);

#ifdef USE_GPU
    t0 = elapsed();
    auto cpu_index = faiss::gpu::index_gpu_to_cpu(&index2);
    t1 = elapsed();
    printf(">>%fs,  gpu to cpu\n", t1 - t0);

    t0 = elapsed();
    cpu_index->remove_ids(sel);
    t1 = elapsed();
    printf(">>%fs,  remove_ids\n", t1 - t0);
    //printf("%d ntotal\n", cpu_index->ntotal);

    t0 = elapsed();
    index2 = *(faiss::IndexIDMap *)faiss::gpu::index_cpu_to_gpu(&res, device, cpu_index, NULL);
    t1 = elapsed();
    printf(">>%fs,  cpu to gpu\n", t1 - t0);

#else
    t0 = elapsed();
    index2.remove_ids(sel);
    t1 = elapsed();
    printf(">>%fs,  remove_ids\n", t1 - t0);
#endif 
    
    //printf("%d ntotal\n", index2.ntotal);


    dis.clear();
    nns.clear();
    t0 = elapsed();
    index2.search(nq, query.data(), k, dis.data(), nns.data());
    t1 = elapsed();
    printf(">>%fs,  search\n", t1 - t0);

    printf("-------- search results --------\n");   
    for(size_t i=0; i < k; ++i)
        printf("%ld   %f\n", nns[i], dis[i]);
    printf("\n-------- search results --------\n");

    return 0;
}
