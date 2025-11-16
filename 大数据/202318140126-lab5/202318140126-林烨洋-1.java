package com.bigdata.hadoop.hdfs;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import java.io.IOException;
public class HDFSUploadFile {
    public void uploadFile(String localFilePath, String hdfsFilePath, boolean isOverwrite) {
        Configuration conf = new Configuration();
        conf.set("fs.defaultFS", "hdfs://localhost:9000"); 
        FileSystem fs = null;
        try {
            fs = FileSystem.get(conf);
            Path localPath = new Path(localFilePath);
            Path hdfsPath = new Path(hdfsFilePath);

            // 检查HDFS中目标文件是否已存在
            if (fs.exists(hdfsPath)) {
                if (isOverwrite) {
                    System.out.println("文件已存在，执行覆盖操作...");
                    fs.delete(hdfsPath, false); // 删除已有文件
                } else {
                    System.out.println("文件已存在，执行跳过操作...");
                    return;
                }
            }

            // 上传本地文件到HDFS
            fs.copyFromLocalFile(localPath, hdfsPath);
            System.out.println("文件上传成功！目标路径：" + hdfsFilePath);
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (fs != null) {
                try {
                    fs.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    public static void main(String[] args) {
        HDFSUploadFile uploader = new HDFSUploadFile();
        uploader.uploadFile(
            "/home/bigdata/test2.txt", 
            "hdfs://localhost:9000/input/test2.txt", 
            true
        );
    }
}