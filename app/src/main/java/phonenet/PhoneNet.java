package phonenet;

import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.util.Log;

/**
 * Created by DELL-PC on 9/5/2017.
 */

public class PhoneNet extends Tensor{
    String tag = "PhoneNet";
    public PhoneNet(){
        super();
    }
    public PhoneNet(RenderScript rs){
        super(rs);
    }
    public boolean setRS(RenderScript rs){
        this.rs = rs;
        return true;
    }
    public PhoneNet setInput(Tensor input){
        this.set(input);
        return this;
    }

    public PhoneNet pool(String type, int kernel_size, int stride)
    {

        String info=type+" Pooling:\n"+"\tInput:"+this.shape.toString();

        int wout = (this.shape.y - kernel_size) / stride + 1;
        int hout = (this.shape.x - kernel_size) / stride + 1;

        ScriptC_conv conv_rs = new ScriptC_conv(rs);
        conv_rs.set_kernel_size(kernel_size);
        conv_rs.set_hin(this.shape.x);
        conv_rs.set_win(this.shape.y);
        conv_rs.set_hout(hout);
        conv_rs.set_wout(wout);
        conv_rs.set_stride(stride);
        Tensor idspace = new Tensor(new Shape(hout,wout,this.shape.z/4),rs);
        Tensor output = new Tensor(new Shape(hout,wout,this.shape.z),rs);
        conv_rs.set_input(this.buff);
        conv_rs.set_output(output.buff);

        if(type=="max")
            conv_rs.forEach_maxPool(idspace.buff);
        else if(type=="avg")
            conv_rs.forEach_avgPool(idspace.buff);
        else
            Log.e(tag,"unknown pooling type: "+type);

        this.set(output);

        info +=" => "+output.shape.toString()+"\n\tStride: "+String.valueOf(stride)+"\tKernel size: "
                +String.valueOf(kernel_size);
        Log.e(tag,info);
        Log.e(tag,"Result:\n"+this.toString(new Shape(2,4,10)));

        return this;

    }

    public PhoneNet conv(Tensor weight,Tensor bias, int stride,int pad){
        /***
         * input_sp : (hin, win, input_channel)
         * weight_sp4: (kernel_size, kernel_size, input_channel4, kernel_num)
         * kernel_sp: (kernel_size, kernel_size, kernel_num)
         * output_sp: (hout, wout, kernel_num)
         * **/

        //(this.shape.x, this.shape.y, this.shape.z ) X (kernel_sp.x, kernel_sp.x, this.shape.z, kernel_sp.z)
        // => output_sp
        String info="Convolution:\n"+"\tInput:"+this.shape.toString()+" x "+weight.shape.toString();

        Shape kernel_sp = new Shape(weight.shape.x,weight.shape.x,weight.shape.w);
        ScriptC_conv conv_rs = new ScriptC_conv(rs);

        int wout = (this.shape.y + 2 * pad - kernel_sp.x) / stride + 1;
        int hout = (this.shape.x + 2 * pad - kernel_sp.x) / stride + 1;

        Shape output_sp = new Shape(hout,wout,kernel_sp.z);
        Tensor output = new Tensor(output_sp,rs);

        conv_rs.set_weight(weight.buff);
        conv_rs.set_bias(bias.buff);
        //the following parameters is automatic filled.
        Tensor idspace = new Tensor(kernel_sp,rs);
        conv_rs.set_kernel_size(kernel_sp.x);
        conv_rs.set_wout(output_sp.y);
        conv_rs.set_hout(output_sp.x);
        conv_rs.set_win(this.shape.y);
        conv_rs.set_hin(this.shape.x);
        conv_rs.set_kernel_num(kernel_sp.z);
        conv_rs.set_input_channel(this.shape.z); //4 or 3?
        conv_rs.set_stride(stride);
        conv_rs.set_pad(pad);
        conv_rs.set_input_channel4(this.shape.z/4);
        conv_rs.set_offset(0);
        conv_rs.set_input(this.buff);
        conv_rs.set_output(output.buff);
        conv_rs.set_parallelOFMs(kernel_sp.z);
        conv_rs.forEach_conv(idspace.buff);


        this.set(output);


        info += " => "+output_sp.toString()+"\n\tStride: "+String.valueOf(stride)+"\tPad: "+
                String.valueOf(pad);
        Log.e(tag,info);
        Log.e(tag,"Result:\n"+this.toString(new Shape(2,4,10)));

        return this;

    }

    public PhoneNet conv(Tensor weight,Tensor bias, int stride,int pad,
                         Tensor weight2, Tensor bias2, int stride2, int pad2){
        /*** This function is just for squeezenet test goal.
         *  pos     : to merge two convolution cell.
         * input_sp : (hin, win, input_channel)
         * weight_sp4: (kernel_size, kernel_size, input_channel4, kernel_num)
         * kernel_sp: (kernel_size, kernel_size, kernel_num)
         * output_sp: (hout, wout, kernel_num)
         * **/

        //(this.shape.x, this.shape.y, this.shape.z ) X (kernel_sp.x, kernel_sp.x, this.shape.z, kernel_sp.z)
        // => output_sp

        String info="Convolution1:\n"+"\tInput:"+this.shape.toString()+" x "+weight.shape.toString();
        String info2="Convolution2:\n"+"\tInput:"+this.shape.toString()+" x "+weight2.shape.toString();

        ScriptC_conv conv_rs = new ScriptC_conv(rs);

        Shape kernel_sp = new Shape(weight.shape.x,weight.shape.x,weight.shape.w);
        int wout = (this.shape.y + 2 * pad - kernel_sp.x) / stride + 1;
        int hout = (this.shape.x + 2 * pad - kernel_sp.x) / stride + 1;
        Shape kernel_sp2 = new Shape(weight2.shape.x,weight2.shape.x,weight2.shape.w);
        int wout2 = (this.shape.y + 2 * pad2 - kernel_sp2.x) / stride2 + 1;
        int hout2 = (this.shape.x + 2 * pad2 - kernel_sp2.x) / stride2 + 1;
        assert wout==wout2 && hout==hout2;

        //the next convoluton will start from wout*hout*kernel1 number;
        int pos = wout * hout * kernel_sp.z;

        Shape output_sp = new Shape(hout,wout,kernel_sp.z+kernel_sp2.z);
        Tensor output = new Tensor(output_sp,rs);

        conv_rs.set_weight(weight.buff);
        conv_rs.set_bias(bias.buff);
        //the following parameters is automatic filled.
        Tensor idspace = new Tensor(kernel_sp,rs);
        conv_rs.set_kernel_size(kernel_sp.x);
        conv_rs.set_wout(output_sp.y);
        conv_rs.set_hout(output_sp.x);
        conv_rs.set_win(this.shape.y);
        conv_rs.set_hin(this.shape.x);
        conv_rs.set_kernel_num(kernel_sp.z);
        conv_rs.set_input_channel(this.shape.z); //4 or 3?
        conv_rs.set_stride(stride);
        conv_rs.set_pad(pad);
        conv_rs.set_input_channel4(this.shape.z/4);
        conv_rs.set_offset(pos);
        conv_rs.set_input(this.buff);
        conv_rs.set_output(output.buff);
        conv_rs.set_parallelOFMs(kernel_sp.z);
        conv_rs.forEach_conv(idspace.buff);

        //we can not call this.set(output) as it will overwrite the input buff.

        conv_rs.set_weight(weight2.buff);
        conv_rs.set_bias(bias2.buff);
        //the following parameters is automatic filled.
        Tensor idspace2 = new Tensor(kernel_sp2,rs);
        conv_rs.set_kernel_size(kernel_sp2.x);
        conv_rs.set_wout(output_sp.y);
        conv_rs.set_hout(output_sp.x);
        conv_rs.set_win(this.shape.y);
        conv_rs.set_hin(this.shape.x);
        conv_rs.set_kernel_num(kernel_sp2.z);
        conv_rs.set_input_channel(this.shape.z); //4 or 3?
        conv_rs.set_stride(stride2);
        conv_rs.set_pad(pad2);
        conv_rs.set_input_channel4(this.shape.z/4);
        conv_rs.set_offset(pos);
        conv_rs.set_input(this.buff);
        conv_rs.set_output(output.buff);
        conv_rs.set_parallelOFMs(kernel_sp2.z);
        conv_rs.forEach_conv(idspace2.buff);
        //now we can overwrite the input buff.
        this.set(output);

        info += " => "+new Shape(hout,wout,kernel_sp.z).toString()+"\n\tStride: "+String.valueOf(stride)+"\tPad: "+
                String.valueOf(pad)+"\t Pos: "+String.valueOf(0);
        Log.e(tag,info);
        info2 += " => "+new Shape(hout,wout,kernel_sp2.z).toString()+"\n\tStride2: "+String.valueOf(stride2)+"\tPad2: "+
                String.valueOf(pad2)+"\t Pos: "+String.valueOf(pos)
                +"\nMerge output: "+new Shape(hout,wout,kernel_sp.z+kernel_sp2.z).toString();
        Log.e(tag,info2);
        Log.e(tag,"Result:\n"+this.toString(new Shape(2,4,10)));

        return this;

    }
    public float[] softmax(){
        float[] in = this.get();

        float[] out = new float[in.length];
        int N = in.length;

        float maxIn, sum;
        int n;
        maxIn = -Float.MAX_VALUE;
        sum = 0;
        //Calculate max input.
        for (n = 0; n<N; n++){
            maxIn = Math.max(maxIn, in[n]);
        }
        //Calculate the exponential of each input.
        for (n = 0; n<N; n++){
            out[n] = (float) Math.exp(in[n] - maxIn);
        }
        //Calculate the sum of all exponentials.
        for (n = 0; n<N; n++){
            sum += out[n];
        }
        //Calculate the output.
        for (n = 0; n<N; n++){
            out[n] = out[n] / sum;
        }

        int maxAt = 0;
        for (int i = 0; i < out.length; i++) {
            maxAt = out[i] > out[maxAt] ? i : maxAt;
        }
        Log.e(tag,"Softmax: max index : "+ String.valueOf(maxAt));

         return out;
    }
}
