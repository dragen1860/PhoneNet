#pragma version(1)
#pragma rs java_package_name(phonenet)


int32_t win;
int32_t hin;
int32_t input_channel;
int32_t input_channel4;

int32_t kernel_size;
int32_t kernel_num;
int32_t stride;
int32_t pad;

int32_t wout;
int32_t hout;

int32_t offset;
int32_t parallelOFMs;

rs_allocation weight;
rs_allocation input;
rs_allocation bias;
rs_allocation output;


void init(){}

float RS_KERNEL conv(uint32_t x) {
/**
* This is general purpose convolution unit. In order to run convolution properly,
* You need to setup some parameters:
* Input side:  input : input buffer
               win   :  width of input
               hin   :  height of input
               input_channel : input channel number, namely input layers number
               input_channel4: input_channel4 = input_channel/4, this is for speedup purpose

* Weight side: weight:  weight buffer
               bias  :  bias buffer
               kernel_size: kernel should have equal height/weight, that is , kernel_size
               kernel_num : kernel number, namely number of filters
               stride:  convolution stride
               pad   :

* output side: wout  : width of output
               hout  : height of output
* IDSpace    : x     : the parallel threads are generated according to IDSpace, we divided each thread
                       by each pixel on output multiply by kernel_num, and x is used to derive current thread
                       situation/position in thread context.
Return:        output: output buffer
**/
     //1. derive current pixel position according to parameter x
	int32_t w = (x / 4) % wout;                         //width idx
	int32_t h = (((x - 4 * w) / 4) / wout) % hout;      //height idx
	int32_t c = (x % 4) + (x / (4 * wout * hout)) * 4;  //channel idx

    //2. on each pixel of output(m,h,w), we have Value(m,h,w) = bias + sum(kernel(around)*input(aroudn))
    // the sum set includes: different kernel number and different kernel position
	float out0 = rsGetElementAt_float(bias, c);

	int32_t n, i, j;
	// for each pixel on (n,i,j) we can derive its related position on input buffer and weight buffer.
	// just fetch its value and accumulate all.
	for (n = 0; n < input_channel4; n++){
		for (i = 0; i < kernel_size; i++)  {
			for (j = 0; j < kernel_size; j++){

				if (w*stride + i - pad<0 || w*stride + i - pad >= win
				     ||h*stride + j - pad<0 || h*stride + j - pad >= hin) continue;

                int32_t in_idx = (w * stride + i - pad) + win * (h * stride + j - pad) + (win * hin * n);
				float4 in0 = rsGetElementAt_float4(input, in_idx);

				int32_t wt_idx = (i) + (kernel_size * j) + (kernel_size * kernel_size * n)
				                    + ((kernel_size * kernel_size * input_channel4) * c);
				float4 wt0 = rsGetElementAt_float4(weight, wt_idx);

				out0 += dot(in0, wt0);

			}
		}
	}

	rsSetElementAt_float(output, fmax(0.0f, out0), offset + x);
}



float RS_KERNEL avgPool(int32_t x) {
/**
* This is general purpose average pooling thread function. To run this thread, you need to setup some
* necessary parameters:
* Input:    hin,win    : height and width of input allocation
*           input      : input allocation
* Kernel:   kernel_size: kernel size
*           stride     : stride
* Output:   hout,wout  : height and width of output allocation
* Return:   output     : output allocation
**/
    int32_t w = (x) % wout;                 //width idx
    int32_t h = ((x) / wout) % hout;        //height idx
    int32_t n = (x / (4 * wout * hout)) * 4;//channel idx, step by 4, such as 0,4,8
    int32_t m = x / (wout * hout);          //actual channel idx, step by 1

    int32_t wstart = w * stride;            //start idx of around area
    int32_t hstart = h * stride;            //start idx of around area
    int32_t wend = wstart + kernel_size;    //end idx of around area
    int32_t hend = hstart + kernel_size;    //end idx of around area

    float4 sum;
    sum.x = 0;
    sum.y = 0;
    sum.z = 0;
    sum.w = 0;
    int32_t ww, hh;
    for (ww = wstart; ww < wend; ww++){ //accumulate around value together.
        for (hh = hstart; hh < hend; hh++){
            sum = sum + rsGetElementAt_float4(input, ww + win * hh + win * hin * m);
        }
    }
    sum = sum / (kernel_size * kernel_size);    //get average pooling value

    int32_t idx = (x % (wout * hout));
    idx = 4 * idx;
    rsSetElementAt_float(output, sum.x, (idx +     wout * hout * 4 * m));
    rsSetElementAt_float(output, sum.y, (idx + 1 + wout * hout * 4 * m));
    rsSetElementAt_float(output, sum.z, (idx + 2 + wout * hout * 4 * m));
    rsSetElementAt_float(output, sum.w, (idx + 3 + wout * hout * 4 * m));

}

float RS_KERNEL maxPool (int32_t x) {
/**
* This is general purpose maximum pooling thread function. To run this thread, you need to setup some
* necessary parameters:
* Input:    hin,win    : height and width of input allocation
*           input      : input allocation
* Kernel:   kernel_size: kernel size
*           stride     : stride
* Output:   hout,wout  : height and width of output allocation
* Return:   output     : output allocation
**/
    int32_t w = (x) % wout;                 //width idx
    int32_t h = ((x) / wout) % hout;        //height idx
    int32_t n = (x / (4 * wout * hout)) * 4;//channel idx, step by 4
    int32_t m = x / (wout * hout);          //actual channel idx, step by 1


    int32_t wstart = w * stride;
    int32_t hstart = h * stride;
    int32_t wend = wstart + kernel_size;
    int32_t hend = hstart + kernel_size;

    int32_t ww, hh;
    //ToDo: Replace this with -MAX_FLT
    float4 maxValue;
    maxValue.x = -3.402823e10;
    maxValue.y = -3.402823e10;
    maxValue.z = -3.402823e10;
    maxValue.w = -3.402823e10;
    for (ww = wstart; ww < wend; ww++){
        for (hh = hstart; hh < hend; hh++){
            maxValue = fmax(maxValue, rsGetElementAt_float4(input, ww + win * hh + win * hin * m));
        }
    }

    int32_t idx = (x % (wout * hout));
    idx = 4 * idx;
    rsSetElementAt_float(output, maxValue.x, (idx +     wout * hout * 4 * m));
    rsSetElementAt_float(output, maxValue.y, (idx + 1 + wout * hout * 4 * m));
    rsSetElementAt_float(output, maxValue.z, (idx + 2 + wout * hout * 4 * m));
    rsSetElementAt_float(output, maxValue.w, (idx + 3 + wout * hout * 4 * m));
}