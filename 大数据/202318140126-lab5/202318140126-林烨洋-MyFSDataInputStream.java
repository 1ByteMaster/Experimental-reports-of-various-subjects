package com.bigdata.hadoop.hdfs;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FSDataInputStream;
import java.io.BufferedReader;
import java.io.InputStreamReader;

public class MyFSDataInputStream {
    public void readHDFSFileByLine() {
        FSDataInputStream in = null;
        BufferedReader br = null;
        try {
            Configuration conf = new Configuration();
            conf.set("fs.defaultFS", "hdfs://localhost:9000"); // 与DFS Locations的node001配置一致
            conf.set("fs.hdfs.impl", "org.apache.hadoop.hdfs.DistributedFileSystem");

            FileSystem fs = FileSystem.get(conf);
            Path path = new Path("/input/test.txt"); // 对应DFS Locations中input目录下的lineyang.txt

            if (!fs.exists(path)) {
                System.out.println("文件不存在: " + path);
                return;
            }

            in = fs.open(path);
            br = new BufferedReader(new InputStreamReader(in));

            System.out.println("文件内容如下:");
            System.out.println("-----------------------------------");
            String line;
            while ((line = br.readLine()) != null) {
                System.out.println(line);
            }
            System.out.println("-----------------------------------");
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                if (br != null) br.close();
                if (in != null) in.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    public static void main(String[] args) {
        MyFSDataInputStream reader = new MyFSDataInputStream();
        reader.readHDFSFileByLine();
    }
}