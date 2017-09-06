package com.openvw.phonenet;

import android.Manifest;
import android.content.DialogInterface;
import android.content.pm.PackageManager;
import android.os.Environment;
import android.renderscript.RenderScript;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;

import phonenet.SqueezeNet;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        permissionCheck();
    }

    @Override
    protected void onStart(){
        super.onStart();

        SqueezeNet2 squeezeNet2 = new SqueezeNet2();
        try{
            squeezeNet2.parSqueezeNet(getApplicationContext());

        }catch (InterruptedException ex){
            Log.e("MainActivity",ex.getMessage());
        }

        SqueezeNet net = new SqueezeNet(RenderScript.create(this));
        net.forward(Environment.getExternalStorageDirectory().getPath()+"/SqueezeNet/Parameters/Vectorized/");

    }

    private void permissionCheck() {
        if (this.checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) !=
                PackageManager.PERMISSION_GRANTED){
            if (shouldShowRequestPermissionRationale(Manifest.permission.WRITE_EXTERNAL_STORAGE)) {
                AlertDialog.Builder builder = new AlertDialog.Builder(this);
                builder.setMessage("It is required to access the SD card to load CNN parameters.");
                builder.setPositiveButton(android.R.string.ok, new DialogInterface.OnClickListener(){
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        requestPermissions(new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},4343);
                    }
                });
                builder.create().show();
            }else{
                requestPermissions(new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},4343);
            }
        }
    }
}
