package phonenet;

import android.os.Environment;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.util.Log;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;

/**
 * Created by DELL-PC on 9/5/2017.
 */

public class Loader {

    String tag = "Loader";
    RenderScript rs=null;
    public Loader(RenderScript rs){
        this.rs = rs;
    }

    public Tensor loadBin(Shape shape, String path){
        float[] data = new float[shape.length()];
        _loadBin(data,path);
        Tensor ts = new Tensor(shape,rs);
        ts.set(data);
        return ts;
    }
    public Tensor loadBin(Shape shape, Element dtype, String path){
        if(dtype==Element.F32_4(rs)){
            float[] data = new float[4*shape.length()];
            _loadBin(data,path);
            Tensor ts = new Tensor(shape, dtype, rs);
            ts.set(data);
            return ts;
        }
        float[] data = new float[shape.length()];
        _loadBin(data,path);
        Tensor ts = new Tensor(shape, dtype, rs);
        ts.set(data);
        return ts;
    }

    private boolean _loadBin(float[] data, String path) {
        try {
            RandomAccessFile file = new RandomAccessFile(path, "rw");
            FileChannel channel = file.getChannel();
            ByteBuffer buf = ByteBuffer.allocate(4 * data.length);
            buf.clear();
            channel.read(buf);
            buf.rewind();
            buf.order(ByteOrder.LITTLE_ENDIAN);
            buf.asFloatBuffer().get(data);
            channel.close();
            file.close();

//            Log.e(tag,path+" loaded!");
        } catch (IOException e) {
            Log.e(tag, e.getMessage());
            return false;
        }
        return true;
    }

    public Tensor SqueezeNet_LoadImgBin(String path){
        Shape sp = new Shape(3,227,227);
        Tensor t = loadBin(sp, path);


        Shape sp2 = new Shape(227,227,4);
        Tensor output = new Tensor(sp2,rs);
        Tensor idspace = new Tensor(new Shape(227,227),rs);
        ScriptC_utility utility_rs = new ScriptC_utility(rs);
        utility_rs.set_hout(227);
        utility_rs.set_wout(227);
        utility_rs.set_input(t.buff);
        utility_rs.set_output(output.buff);
        utility_rs.forEach_channel2f4(idspace.buff);

        return output;
    }
}
