package org.lukey.hadoop.bayes.trainning;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

public class MyUtils {

	// 获取文件夹下面二级文件夹路径的方法
	static List<Path> getSecondDir(Configuration conf, String folder) throws Exception {
		// System.out.println("-2---getSencondDir----" + folder);
		int FILENUMBER = conf.getInt("FILENUMBER", 10);
		Path path = new Path(folder);

		FileSystem fs = path.getFileSystem(conf);
		FileStatus[] stats = fs.listStatus(path);
		System.out.println("stats.length = " + stats.length);
		List<Path> folderPath = new ArrayList<Path>();
		for (FileStatus stat : stats) {
			if (stat.isDir()) {
				// System.out.println("----stat----" + stat.getPath());
				if (fs.listStatus(stat.getPath()).length > FILENUMBER) { // 筛选出文件数大于FILENUMBER的类别作为
					// 输入路径
					folderPath.add(stat.getPath());
				}
			}
		}
		// System.out.println("----folderPath----" + folderPath.size());
		return folderPath;
	}

	// 判断一个字符串是否含有数字
	static boolean hasDigit(String content) {
		boolean flag = false;
		Pattern p = Pattern.compile(".*\\d+.*");
		Matcher m = p.matcher(content);
		if (m.matches())
			flag = true;
		return flag;
	}

	// 读取先验概率和类别中没有出现单词的概率
	static Map<String, Double> getProbability(Configuration conf, String path) throws IOException {
		Map<String, Double> pMap = new HashMap<>();
		FileSystem fs = null;
		fs = FileSystem.get(URI.create(path), conf);// 获取文件系统对象
		FSDataInputStream inputStream = fs.open(new Path(path));// 定义文件输入流
		BufferedReader buffer = new BufferedReader(new InputStreamReader(inputStream));// 定义读缓冲区
		String strLine = null;
		while ((strLine = buffer.readLine()) != null) {// 按行读取
			String[] temp = strLine.split("\t");
			pMap.put(temp[0], Double.parseDouble(temp[1])); // 得到概率保存到map中
		}
		return pMap;
	}
	
	// 获取条件概率方法
	static Map<String, Map<String, Double>> getConditionMap(Configuration conf, String dirPath) throws Exception {
		Map<String, Map<String, Double>> condMap = new HashMap<>();
		Path dir = new Path(dirPath);
		FileSystem fs = dir.getFileSystem(conf);
		String className = "";
		for (FileStatus file : fs.listStatus(dir)) {
			if (!file.isDir()) {//是文件时继续执行
				Path filePath = file.getPath();// 获取文件路径
				String fileName = filePath.getName();// 获取文件名ALB-m-00000
				String[] temp = fileName.split("-");//提取出类别名
				if (temp.length == 3) {
					className = temp[0];// 得到类别名
				}
				Map<String, Double> oneMap = MyUtils.getProbability(conf, filePath.toString());
				condMap.put(className, oneMap);
			}
		}
		return condMap;
	}
	
	
}
