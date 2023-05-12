#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include "compute.h"
//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
//we shall define 3 functions here


__global__ void arraySet(vector3* v, vector3** a){
        int in = threadIdx.x;
        //for (int i=in;i<NUMENTITIES;i += blockDim.x){
                if(in < NUMENTITIES){
                        a[in]=&v[in*NUMENTITIES];
                }
        //}
}

__global__ void accelComp(vector3** a, vector3* pos, double* mass){
        int in = threadIdx.x;
        int j,k;
        //for (i=in;i<NUMENTITIES;i += blockDim.x){
        if(in < NUMENTITIES){
                for (j=0;j<NUMENTITIES;j++){
                        if (in==j) {
                                FILL_VECTOR(a[in][j],0,0,0);
                        }
                        else{
                                vector3 distance;
                                for (k=0;k<3;k++) distance[k]=pos[in][k]-pos[j][k];
                                double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
                                double magnitude=sqrt(magnitude_sq);
                                double accelmag=-1*GRAV_CONSTANT*mass[j]/magnitude_sq;
                                FILL_VECTOR(a[in][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
                        }
                }
        }
}

__global__ void sumMatrix(vector3** a, vector3* pos, vector3* vel){
        int j,k;
        int in = threadIdx.x;
        //for (i=in;i<NUMENTITIES;i+= blockDim.x){
        if(in < NUMENTITIES){
                vector3 accel_sum={0,0,0};
                if(in < NUMENTITIES){
                for (j=0;j<NUMENTITIES;j++){
                        for (k=0;k<3;k++)
                                accel_sum[k]+=a[in][j][k];
                }
                //compute the new velocity based on the acceleration and time interval
                //compute the new position based on the velocity and time interval
                for (k=0;k<3;k++){
                        vel[in][k]+=accel_sum[k]*INTERVAL;
                        pos[in][k]=vel[in][k]*INTERVAL;
                }
        }}
}
void compute(){
        //make an acceleration matrix which is NUMENTITIES squared in size;
        vector3* values;
        vector3** accels;
        double* d_mass;
        //device memory start
        //vector3* values=(vector3*)malloc(sizeof(vector3)*NUMENTITIES*NUMENTITIES);
        //vector3** accels=(vector3**)malloc(sizeof(vector3*)*NUMENTITIES);
        cudaMalloc((void**)&values,sizeof(vector3)*NUMENTITIES*NUMENTITIES);
        cudaMalloc((void**)&accels,sizeof(vector3)*NUMENTITIES);
        cudaMalloc((void**)&d_mass,sizeof(double));
        cudaMalloc((void**)&d_hPos,sizeof(vector3)*NUMENTITIES);
        cudaMalloc((void**)&d_hVel,sizeof(vector3)*NUMENTITIES);
        cudaMemcpy(d_hPos,hPos,sizeof(vector3)*NUMENTITIES,cudaMemcpyHostToDevice);
        cudaMemcpy(d_hVel,hVel,sizeof(vector3)*NUMENTITIES,cudaMemcpyHostToDevice);
        cudaMemcpy(d_mass,mass,sizeof(double),cudaMemcpyHostToDevice);
        //for (i=0;i<NUMENTITIES;i++)
        //accels[i]=&values[i*NUMENTITIES];
        arraySet<<<1,1008>>>(values,accels);
        //cudaCheckError();
        cudaDeviceSynchronize();
        //first compute the pairwise accelerations.  Effect is on the first argument.
        accelComp<<<1,1008>>>(accels,d_hPos,d_mass);
        cudaDeviceSynchronize();
        //sum up the rows of our matrix to get effect on each entity, then update velocity and position.
        sumMatrix<<<1,1008>>>(accels,d_hPos,d_hVel);
        cudaDeviceSynchronize();
        //free(accels);
        //free(values);
        //free device memory
        cudaMemcpy(hPos,d_hPos,sizeof(vector3)*NUMENTITIES,cudaMemcpyDeviceToHost);
        cudaMemcpy(hVel,d_hVel,sizeof(vector3)*NUMENTITIES,cudaMemcpyDeviceToHost);
        cudaMemcpy(mass,d_mass,sizeof(double),cudaMemcpyDeviceToHost);
        cudaFree(accels);
        cudaFree(values);
        cudaFree(d_mass);
        cudaFree(d_hPos);
        cudaFree(d_hVel);
}
