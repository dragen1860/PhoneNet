#pragma version(1)
#pragma rs java_package_name(phonenet)

int wout,hout;
rs_allocation input,output;


void init(){}

float RS_KERNEL channel2f4(uint32_t x) {
    //Wout = Hout = 227
    //exe space : 227 * 227
    int32_t w = x % wout;
    int32_t h = ((x - w) / wout) % hout;

    //[channel,height,width] => [height,width] and each element contains [r,g,b,0]
    float4 out;
    out.x = rsGetElementAt_float(input, w + wout * h);
    out.y = rsGetElementAt_float(input, w + wout * h + (wout * hout * 1));
    out.z = rsGetElementAt_float(input, w + wout * h + (wout * hout * 2));
    out.w = 0;

    rsSetElementAt_float4(output, out, x);
}