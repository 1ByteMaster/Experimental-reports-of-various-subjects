package com.bigdata.hadoop.hdfs;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import java.io.IOException;
public class HDFSDirectoryCreator {

    public void createDirectory(String hdfsDirPath) {
        Configuration conf = new Configuration();
        conf.set("fs.defaultFS", "hdfs://localhost:9000"); 
        FileSystem fs = null;
        try {
            fs = FileSystem.get(conf);
            Path dirPath = new Path(hdfsDirPath);
            // 调用mkdirs()方法，自动创建所有不存在的上层目录
            boolean isCreated = fs.mkdirs(dirPath);
            if (isCreated) {
                System.out.println("目录创建成功：" + hdfsDirPath);
            } else {
                System.out.println("目录已存在或创建失败：" + hdfsDirPath);
            }
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
        HDFSDirectoryCreator creator = new HDFSDirectoryCreator();
        // 示例：创建HDFS目录 /user/bigdata/testDir（自动创建上层不存在的目录）
        creator.createDirectory("hdfs://localhost:9000/user/bigdata/testDir");
    }
}