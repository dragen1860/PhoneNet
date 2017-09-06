#pragma version(1)
#pragma rs java_package_name(phonenet)

int lenght;
int index;
float data;

void init(){}

void tensor_max(rs_allocation buff,int length){
    int i=0;
    index=0;
    data = rsGetElementAt_float(buff,0);
    for(;i<length;i++)
    {
        float value = rsGetElementAt_float(buff,i);
        if(value > data){
            index = i;
            data = value;
        }
    }
}


