package com.bigdata.hadoop.hdfs;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import java.io.BufferedReader;
import java.io.InputStreamReader;

public class HDFSFileReader {
    public void readHDFSFileToTerminal(String hdfsFilePath) {
        // 声明需要关闭的资源
        FileSystem fs = null;
        FSDataInputStream in = null;
        BufferedReader br = null;
        try {
            // 1. 创建Hadoop配置对象，设置HDFS连接信息
            Configuration conf = new Configuration();
            conf.set("fs.defaultFS", "hdfs://localhost:9000");
            conf.set("fs.hdfs.impl", "org.apache.hadoop.hdfs.DistributedFileSystem");
            // 2. 获取HDFS文件系统实例
            fs = FileSystem.get(conf);
            // 3. 检查文件是否存在
            Path path = new Path(hdfsFilePath);
            if (!fs.exists(path)) {
                System.out.println("错误：HDFS中不存在该文件 - " + hdfsFilePath);
                return;
            }

            // 4. 打开文件输入流
            in = fs.open(path);

            // 5. 包装为BufferedReader便于按行读取
            br = new BufferedReader(new InputStreamReader(in));

            // 6. 读取内容并输出到终端
            System.out.println("===== HDFS文件内容开始 =====");
            String line;
            while ((line = br.readLine()) != null) {
                System.out.println(line); // 输出到终端
            }
            System.out.println("===== HDFS文件内容结束 =====");

        } catch (Exception e) {
            System.err.println("读取文件时发生错误：");
            e.printStackTrace();
        } finally {
            // 7. 关闭所有资源
            try {
                if (br != null) br.close();
                if (in != null) in.close();
                if (fs != null) fs.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    public static void main(String[] args) {
        // 实例化文件读取器
        HDFSFileReader reader = new HDFSFileReader();
        
        // 读取HDFS中的指定文件并输出到终端
        String hdfsFilePath = "hdfs://localhost:9000/input/test2.txt";
        reader.readHDFSFileToTerminal(hdfsFilePath);
    }
}
