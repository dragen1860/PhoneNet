package phonenet;

/**
 * Created by DELL-PC on 9/5/2017.
 */

public class Shape {


    public int x,y,z,w;
    public int dim;
    public int shape[];

    public Shape(){
        this.dim = 0;
        shape = null;
    }
    public Shape(int x){
        this.x = x;
        this.y = 0;
        this.z = 0;
        this.w = 0;
        this.dim = 1;
        this.shape = new int[4];
        this.shape[0] = this.x;
        this.shape[1] = this.y;
        this.shape[2] = this.z;
        this.shape[3] = this.w;

    }
    public Shape(int x, int y){
        this.x = x;
        this.y = y;
        this.z = 0;
        this.w = 0;
        this.dim =2;
        this.shape = new int[4];
        this.shape[0] = this.x;
        this.shape[1] = this.y;
        this.shape[2] = this.z;
        this.shape[3] = this.w;

    }

    public Shape(int x, int y, int z){
        this.x = x;
        this.y = y;
        this.z = z;
        this.w = 0;
        this.dim =3;
        this.shape = new int[4];
        this.shape[0] = this.x;
        this.shape[1] = this.y;
        this.shape[2] = this.z;
        this.shape[3] = this.w;

    }
    public Shape(int x, int y, int z, int w){
        this.x = x;
        this.y = y;
        this.z = z;
        this.w = w;
        this.dim =4;
        this.shape = new int[4];
        this.shape[0] = this.x;
        this.shape[1] = this.y;
        this.shape[2] = this.z;
        this.shape[3] = this.w;
    }
    public void set(int x,int y,int z){
        this.x = x;
        this.y = y;
        this.z = z;
        this.w = 0;
        this.dim = 3;
        this.shape = new int[4];
        this.shape[0] = this.x;
        this.shape[1] = this.y;
        this.shape[2] = this.z;
        this.shape[3] = this.w;
    }
    public void set(int x,int y,int z, int w){
        this.x = x;
        this.y = y;
        this.z = z;
        this.w = w;
        this.dim =4;
        this.shape = new int[4];
        this.shape[0] = this.x;
        this.shape[1] = this.y;
        this.shape[2] = this.z;
        this.shape[3] = this.w;
    }

    public void set(int[] size){
        assert  size[0] != 0;

        this.x = size[0];
        this.y = size[1];
        this.z = size[2];
        this.w = size[3];
        this.dim = size[1]!=0?(size[2]!=0?(size[3]!=0?4:3):2):1;
        this.shape = new int[4];
        this.shape[0] = this.x;
        this.shape[1] = this.y;
        this.shape[2] = this.z;
        this.shape[3] = this.w;
    }

    public Shape len2shape(int len){
        int i = dim;
        int[] size=new int[]{0,0,0,0};
        int tmp =1;
        while(i>0){
            tmp *= shape[i];
            size[i] = len % tmp;
        }
        Shape sp = new Shape();
        sp.set(size);
        return sp;
    }

    public int length(){
        if(dim==0)return 0;

        int length = 1;
        for(int i=0;i<dim;i++)length*=shape[i];
        return length;
    }
    public String toString(){
        String buff="(";
        for(int i=0;i<dim;i++)buff+=String.valueOf(shape[i])+",";
        buff+=")";
        return buff;
    }
}
