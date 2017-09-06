package phonenet;

import phonenet.Shape;

import android.graphics.Bitmap;
import android.renderscript.*;

/**
 * Created by DELL-PC on 9/5/2017.
 */

public class Tensor {
    RenderScript rs;
    Allocation buff;
    Shape shape;

    public Tensor(){
        buff = null;
        rs = null;
        shape = new Shape();
    }
    public Tensor(RenderScript rs){
        buff = null;
        shape = new Shape();
        this.rs = rs;
    }
    public Tensor(Shape shape, Element dtype, RenderScript rs){
        this.rs =rs;

        int length = 1;
        for(int i=0;i<shape.dim;i++)length*=shape.shape[i];

        buff = Allocation.createSized(rs, dtype,length);
        this.shape = shape;
    }
    public Tensor(Shape shape, RenderScript rs){
        this.rs = rs;

        int length = 1;
        for(int i=0;i<shape.dim;i++)length*=shape.shape[i];

        buff = Allocation.createSized(rs, Element.F32(rs),length);
        this.shape = shape;

    }

    public boolean set(Tensor ts){
        this.destroy();

        this.buff = ts.buff;
        this.shape = ts.shape;
        return true;
    }

    public boolean reshape(Shape shape){
        int length = 1;
        for(int i=0;i<shape.dim;i++)length*=shape.shape[i];

        assert length==this.length();

        this.shape = shape;
        return true;
    }

    public boolean transpose(Shape shape){
        return true;
    }

    public boolean set(float[] data){
        assert data.length == this.length();

        this.buff.copyFrom(data);
        return true;
    }

    public int length(){
        if(buff==null)return 0;

        int length = 1;
        for(int i=0;i<shape.dim;i++)length*=shape.shape[i];
        return length;

    }

    public boolean set(Bitmap bm){

        return true;
    }

    public Tensor cat(Tensor a, Tensor b, int axis){
        Tensor output = new Tensor();

        return output;
    }

    public float[] get(){

        float data[] = new float[this.length()];
        buff.copyTo(data);
        return data;
    }
    public Tensor getTensor(){
        return this;
    }

    public boolean destroy(){
        if(buff!=null)buff.destroy();
        return true;
    }
    public Shape max(){
        ScriptC_tensor tensor_rs = new ScriptC_tensor(rs);
        tensor_rs.invoke_tensor_max(buff,length());
        float data = tensor_rs.get_data();
        int index = tensor_rs.get_index();

        return shape.len2shape(index);
    }
    public String toString(Shape print_sp){
        String str = new String();
        float[] data = this.get();
        //on case the print_sp is out of bound of shape, we need to join the smaller value.
        int[] tmp = new int[4];
        for(int i=0;i<shape.dim;i++){
            tmp[i] = print_sp.shape[i] > shape.shape[i] ? shape.shape[i] : print_sp.shape[i];
        }
        print_sp.set(tmp);

        if(shape.dim == 1)
        {
            for(int i=0;i<print_sp.x;i++) {
                str+=String.valueOf(data[i]+" ");
            }
            return str;
        }
        else if(shape.dim==2){
            for(int i=0;i<print_sp.x;i++){
                for(int j=0;j<print_sp.y;j++){
                    str+=String.valueOf(data[i*shape.x+j]+" ");
                }
                str+="\n";
            }
            return str;

        }
        else if(shape.dim==3){
            for(int i=0;i<print_sp.x;i++)
            {
                str+=String.format("[%d,:,:]\n",i);
                for(int j=0;j<print_sp.y;j++){
                    for(int k=0;k<print_sp.z;k++){
                        str+=String.format("%3.3f\t",data[i*shape.z*shape.y+j*shape.z+k]);
                    }
                    str+="\n";
                }
            }
            return str;
        }
        else{
            for(int i=0;i<this.length();i++)str+=String.valueOf(data[i])+" ";
            return str;
        }

    }
}
