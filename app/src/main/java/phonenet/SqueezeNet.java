package phonenet;

import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.util.Log;

/**
 * Created by DELL-PC on 9/5/2017.
 */

public class SqueezeNet {

    private Tensor img;
    private Tensor conv1_w;
    private Tensor conv1_b;
    private Tensor conv1_w2;
    private Tensor conv1_b2;


    private RenderScript rs;
    private String tag = "SqueezeNet";
    Shape weight_sp4;
    Shape weight_sp42;

    public SqueezeNet(RenderScript rs){
        this.rs = rs;

    }


    public boolean forward(String dir){
        Tensor input = new Loader(rs).SqueezeNet_LoadImgBin(dir+"img.bin");
        PhoneNet net = new PhoneNet(rs);
        PhoneNet net2 = new PhoneNet(rs);
        net = net.setInput(input);

        weight_sp4 = new Shape(7,7,1,96);     //weight_sp4.z = weight_sp.z /4
        conv1_w = new Loader(rs).loadBin(weight_sp4, Element.F32_4(rs), dir+"conv1_w.bin");
        conv1_b = new Loader(rs).loadBin(new Shape(weight_sp4.w), dir+"conv1_b.bin");
        net = net.conv(conv1_w,conv1_b,2,0);

        net = net.pool("max",3,2);


        weight_sp4 = new Shape(1,1,24,16);     //weight_sp4.z = weight_sp.z /4
        conv1_w = new Loader(rs).loadBin(weight_sp4, Element.F32_4(rs), dir+"fire2_squeeze1x1_w.bin");
        conv1_b = new Loader(rs).loadBin(new Shape(weight_sp4.w), dir+"fire2_squeeze1x1_b.bin");
        net = net.conv(conv1_w,conv1_b,1,0);


        weight_sp4 = new Shape(1,1,4,64);     //weight_sp4.z = weight_sp.z /4
        conv1_w = new Loader(rs).loadBin(weight_sp4, Element.F32_4(rs), dir+"fire2_expand1x1_w.bin");
        conv1_b = new Loader(rs).loadBin(new Shape(weight_sp4.w), dir+"fire2_expand1x1_b.bin");
        weight_sp42 = new Shape(3,3,4,64);     //weight_sp4.z = weight_sp.z /4
        conv1_w2 = new Loader(rs).loadBin(weight_sp42, Element.F32_4(rs), dir+"fire2_expand3x3_w.bin");
        conv1_b2 = new Loader(rs).loadBin(new Shape(weight_sp42.w), dir+"fire2_expand3x3_b.bin");
        net = net.conv(conv1_w,conv1_b,1,0,conv1_w2,conv1_b2,1,1);



        weight_sp4 = new Shape(1,1,32,16);     //weight_sp4.z = weight_sp.z /4
        conv1_w = new Loader(rs).loadBin(weight_sp4, Element.F32_4(rs), dir+"fire3_squeeze1x1_w.bin");
        conv1_b = new Loader(rs).loadBin(new Shape(weight_sp4.w), dir+"fire3_squeeze1x1_b.bin");
        net = net.conv(conv1_w,conv1_b,1,0);


        weight_sp4 = new Shape(1,1,4,64);     //weight_sp4.z = weight_sp.z /4
        conv1_w = new Loader(rs).loadBin(weight_sp4, Element.F32_4(rs), dir+"fire3_expand1x1_w.bin");
        conv1_b = new Loader(rs).loadBin(new Shape(weight_sp4.w), dir+"fire3_expand1x1_b.bin");
        weight_sp42 = new Shape(3,3,4,64);     //weight_sp4.z = weight_sp.z /4
        conv1_w2 = new Loader(rs).loadBin(weight_sp42, Element.F32_4(rs), dir+"fire3_expand3x3_w.bin");
        conv1_b2 = new Loader(rs).loadBin(new Shape(weight_sp42.w), dir+"fire3_expand3x3_b.bin");
        net = net.conv(conv1_w,conv1_b,1,0,conv1_w2,conv1_b2,1,1);


        weight_sp4 = new Shape(1,1,32,32);     //weight_sp4.z = weight_sp.z /4
        conv1_w = new Loader(rs).loadBin(weight_sp4, Element.F32_4(rs), dir+"fire4_squeeze1x1_w.bin");
        conv1_b = new Loader(rs).loadBin(new Shape(weight_sp4.w), dir+"fire4_squeeze1x1_b.bin");
        net = net.conv(conv1_w,conv1_b,1,0);



        weight_sp4 = new Shape(1,1,8,128);     //weight_sp4.z = weight_sp.z /4
        conv1_w = new Loader(rs).loadBin(weight_sp4, Element.F32_4(rs), dir+"fire4_expand1x1_w.bin");
        conv1_b = new Loader(rs).loadBin(new Shape(weight_sp4.w), dir+"fire4_expand1x1_b.bin");
        weight_sp42 = new Shape(3,3,8,128);     //weight_sp4.z = weight_sp.z /4
        conv1_w2 = new Loader(rs).loadBin(weight_sp42, Element.F32_4(rs), dir+"fire4_expand3x3_w.bin");
        conv1_b2 = new Loader(rs).loadBin(new Shape(weight_sp42.w), dir+"fire4_expand3x3_b.bin");
        net = net.conv(conv1_w,conv1_b,1,0,conv1_w2,conv1_b2,1,1);


        net = net.pool("max",3,2);

        weight_sp4 = new Shape(1,1,64,32);     //weight_sp4.z = weight_sp.z /4
        conv1_w = new Loader(rs).loadBin(weight_sp4, Element.F32_4(rs), dir+"fire5_squeeze1x1_w.bin");
        conv1_b = new Loader(rs).loadBin(new Shape(weight_sp4.w), dir+"fire5_squeeze1x1_b.bin");
        net = net.conv(conv1_w,conv1_b,1,0);



        weight_sp4 = new Shape(1,1,8,128);     //weight_sp4.z = weight_sp.z /4
        conv1_w = new Loader(rs).loadBin(weight_sp4, Element.F32_4(rs), dir+"fire5_expand1x1_w.bin");
        conv1_b = new Loader(rs).loadBin(new Shape(weight_sp4.w), dir+"fire5_expand1x1_b.bin");
        weight_sp42 = new Shape(3,3,8,128);     //weight_sp4.z = weight_sp.z /4
        conv1_w2 = new Loader(rs).loadBin(weight_sp42, Element.F32_4(rs), dir+"fire5_expand3x3_w.bin");
        conv1_b2 = new Loader(rs).loadBin(new Shape(weight_sp42.w), dir+"fire5_expand3x3_b.bin");
        net = net.conv(conv1_w,conv1_b,1,0,conv1_w2,conv1_b2,1,1);


        weight_sp4 = new Shape(1,1,64,48);     //weight_sp4.z = weight_sp.z /4
        conv1_w = new Loader(rs).loadBin(weight_sp4, Element.F32_4(rs), dir+"fire6_squeeze1x1_w.bin");
        conv1_b = new Loader(rs).loadBin(new Shape(weight_sp4.w), dir+"fire6_squeeze1x1_b.bin");
        net = net.conv(conv1_w,conv1_b,1,0);



        weight_sp4 = new Shape(1,1,12,192);     //weight_sp4.z = weight_sp.z /4
        conv1_w = new Loader(rs).loadBin(weight_sp4, Element.F32_4(rs), dir+"fire6_expand1x1_w.bin");
        conv1_b = new Loader(rs).loadBin(new Shape(weight_sp4.w), dir+"fire6_expand1x1_b.bin");
        weight_sp42 = new Shape(3,3,12,192);     //weight_sp4.z = weight_sp.z /4
        conv1_w2 = new Loader(rs).loadBin(weight_sp42, Element.F32_4(rs), dir+"fire6_expand3x3_w.bin");
        conv1_b2 = new Loader(rs).loadBin(new Shape(weight_sp42.w), dir+"fire6_expand3x3_b.bin");
        net = net.conv(conv1_w,conv1_b,1,0,conv1_w2,conv1_b2,1,1);


        weight_sp4 = new Shape(1,1,96,48);     //weight_sp4.z = weight_sp.z /4
        conv1_w = new Loader(rs).loadBin(weight_sp4, Element.F32_4(rs), dir+"fire7_squeeze1x1_w.bin");
        conv1_b = new Loader(rs).loadBin(new Shape(weight_sp4.w), dir+"fire7_squeeze1x1_b.bin");
        net = net.conv(conv1_w,conv1_b,1,0);



        weight_sp4 = new Shape(1,1,12,192);     //weight_sp4.z = weight_sp.z /4
        conv1_w = new Loader(rs).loadBin(weight_sp4, Element.F32_4(rs), dir+"fire7_expand1x1_w.bin");
        conv1_b = new Loader(rs).loadBin(new Shape(weight_sp4.w), dir+"fire7_expand1x1_b.bin");
        weight_sp42 = new Shape(3,3,12,192);     //weight_sp4.z = weight_sp.z /4
        conv1_w2 = new Loader(rs).loadBin(weight_sp42, Element.F32_4(rs), dir+"fire7_expand3x3_w.bin");
        conv1_b2 = new Loader(rs).loadBin(new Shape(weight_sp42.w), dir+"fire7_expand3x3_b.bin");
        net = net.conv(conv1_w,conv1_b,1,0,conv1_w2,conv1_b2,1,1);



        weight_sp4 = new Shape(1,1,96,64);     //weight_sp4.z = weight_sp.z /4
        conv1_w = new Loader(rs).loadBin(weight_sp4, Element.F32_4(rs), dir+"fire8_squeeze1x1_w.bin");
        conv1_b = new Loader(rs).loadBin(new Shape(weight_sp4.w), dir+"fire8_squeeze1x1_b.bin");
        net = net.conv(conv1_w,conv1_b,1,0);



        weight_sp4 = new Shape(1,1,16,256);     //weight_sp4.z = weight_sp.z /4
        conv1_w = new Loader(rs).loadBin(weight_sp4, Element.F32_4(rs), dir+"fire8_expand1x1_w.bin");
        conv1_b = new Loader(rs).loadBin(new Shape(weight_sp4.w), dir+"fire8_expand1x1_b.bin");
        weight_sp42 = new Shape(3,3,16,256);     //weight_sp4.z = weight_sp.z /4
        conv1_w2 = new Loader(rs).loadBin(weight_sp42, Element.F32_4(rs), dir+"fire8_expand3x3_w.bin");
        conv1_b2 = new Loader(rs).loadBin(new Shape(weight_sp42.w), dir+"fire8_expand3x3_b.bin");
        net = net.conv(conv1_w,conv1_b,1,0,conv1_w2,conv1_b2,1,1);

        net = net.pool("max",3,2);



        weight_sp4 = new Shape(1,1,128,64);     //weight_sp4.z = weight_sp.z /4
        conv1_w = new Loader(rs).loadBin(weight_sp4, Element.F32_4(rs), dir+"fire9_squeeze1x1_w.bin");
        conv1_b = new Loader(rs).loadBin(new Shape(weight_sp4.w), dir+"fire9_squeeze1x1_b.bin");
        net = net.conv(conv1_w,conv1_b,1,0);




        weight_sp4 = new Shape(1,1,16,256);     //weight_sp4.z = weight_sp.z /4
        conv1_w = new Loader(rs).loadBin(weight_sp4, Element.F32_4(rs), dir+"fire9_expand1x1_w.bin");
        conv1_b = new Loader(rs).loadBin(new Shape(weight_sp4.w), dir+"fire9_expand1x1_b.bin");
        weight_sp42 = new Shape(3,3,16,256);     //weight_sp4.z = weight_sp.z /4
        conv1_w2 = new Loader(rs).loadBin(weight_sp42, Element.F32_4(rs), dir+"fire9_expand3x3_w.bin");
        conv1_b2 = new Loader(rs).loadBin(new Shape(weight_sp42.w), dir+"fire9_expand3x3_b.bin");
        net = net.conv(conv1_w,conv1_b,1,0,conv1_w2,conv1_b2,1,1);



        weight_sp4 = new Shape(1,1,128,1000);     //weight_sp4.z = weight_sp.z /4
        conv1_w = new Loader(rs).loadBin(weight_sp4, Element.F32_4(rs), dir+"conv10_w.bin");
        conv1_b = new Loader(rs).loadBin(new Shape(weight_sp4.w), dir+"conv10_b.bin");
        net = net.conv(conv1_w,conv1_b,1,1);

        net = net.pool("avg",15,1);

        net.softmax();

        return true;
    }
}
