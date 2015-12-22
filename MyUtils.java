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

	// ��ȡ�ļ�����������ļ���·���ķ���
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
				if (fs.listStatus(stat.getPath()).length > FILENUMBER) { // ɸѡ���ļ�������FILENUMBER�������Ϊ
					// ����·��
					folderPath.add(stat.getPath());
				}
			}
		}
		// System.out.println("----folderPath----" + folderPath.size());
		return folderPath;
	}

	// �ж�һ���ַ����Ƿ�������
	static boolean hasDigit(String content) {
		boolean flag = false;
		Pattern p = Pattern.compile(".*\\d+.*");
		Matcher m = p.matcher(content);
		if (m.matches())
			flag = true;
		return flag;
	}

	// ��ȡ������ʺ������û�г��ֵ��ʵĸ���
	static Map<String, Double> getProbability(Configuration conf, String path) throws IOException {
		Map<String, Double> pMap = new HashMap<>();
		FileSystem fs = null;
		fs = FileSystem.get(URI.create(path), conf);// ��ȡ�ļ�ϵͳ����
		FSDataInputStream inputStream = fs.open(new Path(path));// �����ļ�������
		BufferedReader buffer = new BufferedReader(new InputStreamReader(inputStream));// �����������
		String strLine = null;
		while ((strLine = buffer.readLine()) != null) {// ���ж�ȡ
			String[] temp = strLine.split("\t");
			pMap.put(temp[0], Double.parseDouble(temp[1])); // �õ����ʱ��浽map��
		}
		return pMap;
	}
	
	// ��ȡ�������ʷ���
	static Map<String, Map<String, Double>> getConditionMap(Configuration conf, String dirPath) throws Exception {
		Map<String, Map<String, Double>> condMap = new HashMap<>();
		Path dir = new Path(dirPath);
		FileSystem fs = dir.getFileSystem(conf);
		String className = "";
		for (FileStatus file : fs.listStatus(dir)) {
			if (!file.isDir()) {//���ļ�ʱ����ִ��
				Path filePath = file.getPath();// ��ȡ�ļ�·��
				String fileName = filePath.getName();// ��ȡ�ļ���ALB-m-00000
				String[] temp = fileName.split("-");//��ȡ�������
				if (temp.length == 3) {
					className = temp[0];// �õ������
				}
				Map<String, Double> oneMap = MyUtils.getProbability(conf, filePath.toString());
				condMap.put(className, oneMap);
			}
		}
		return condMap;
	}
	
	
}
