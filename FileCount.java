package org.lukey.hadoop.bayes.trainning;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;

/**
 * 
 * 获取文件个数，并计算先验概率 先验概率保存在/user/hadoop/output/priorP/prior.txt
 * 
 */

public class FileCount {

	public static void run(Configuration conf) throws Exception {
		
		String inputFolder = conf.get("input");
		int FILENUMBER = conf.getInt("FILENUMBER", 10);
		
		Map<String, Integer> map = new HashMap<>(); // 保存的是类别和文本数
		Map<String, Double> priorMap = new HashMap<>();// 保存每个类别的先验概率

		// 获取每个类别的文件个数（需要筛选测试集，有的类别文本数太少要删除）
		map = FileCount.getFileNumber(inputFolder);

		// 测试打印出每个类别和文件总数，删除出文本数少于10的类别
		int sum = 0;
		Iterator<Map.Entry<String, Integer>> itrs = map.entrySet().iterator();
		while (itrs.hasNext()) {
			// System.out.println("ok");
			Map.Entry<String, Integer> it = itrs.next();
			if (it.getValue() <= FILENUMBER) { // 这两行代码可以不计算文本数少于FILENUMBER的类别
				itrs.remove();
			} else {
				sum += it.getValue();
				System.out.println(it.getKey() + "\t" + it.getValue());
			}
		}
		System.out.println("所有类别的总数："+ map.size());
		System.out.println("所有类别文本个数总和 ： " + sum);

		// 先验概率的保存路径
		String output = conf.get("priorProbality");
		Path outputPath = new Path(output);
		FileSystem fs = outputPath.getFileSystem(conf);
		FSDataOutputStream outputStream = fs.create(outputPath);

		// 计算 每个类别文本占总文本的比率，即先验概率
		// 写入文件保存先验概率
		String ctx = "";
		for (Map.Entry<String, Integer> entry : map.entrySet()) {
			Double result = 0.0;
			result = Double.parseDouble(entry.getValue().toString()) / sum;
			priorMap.put(entry.getKey(), result);// 保存在priorMap中
			ctx += entry.getKey() + "\t" + result + "\n";
		}
		outputStream.writeBytes(ctx);
		IOUtils.closeStream(outputStream);
		/*
		 * // 打印概率信息，同时可以写入文件中 // map的另外一种遍历方法 Iterator<Map.Entry<String,
		 * Double>> iterators = priorMap.entrySet().iterator(); while
		 * (iterators.hasNext()) { Map.Entry<String, Double> iterator =
		 * iterators.next(); System.out.println(iterator.getKey() + "\t" +
		 * iterator.getValue()); }
		 */

	}

	// 获取文件个数的方法
	public static Map<String, Integer> getFileNumber(String folderPath) throws Exception {

		Map<String, Integer> fileMap = new HashMap<>();
		Configuration conf = new Configuration();

		Path path = new Path(folderPath);
		FileSystem hdfs = path.getFileSystem(conf);
		FileStatus[] status = hdfs.listStatus(path);
		// System.out.println(folderPath);
		// System.out.println("status.length = " + status.length);
		for (FileStatus stat : status) {
			if (stat.isDir()) {
				int length = hdfs.listStatus(stat.getPath()).length;
				String name = stat.getPath().getName();
				fileMap.put(name, length);
			}
		}
		return fileMap;
	}
}

