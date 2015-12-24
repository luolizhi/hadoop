package org.lukey.hadoop.bayes.trainning;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.ArrayList;
import java.util.Set;
import java.util.TreeSet;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

/*
 * 测试集用了单个类别，分别是USA 50个，INDIA 20个， ARG 15个。分别计算其tp fp  fn tn
 */
public class Evaluation {

	public static void run(Configuration conf) throws Exception {
		String input = conf.get("result");

		// 读取评估类别，计算P R F1
		FileSystem fs = FileSystem.get(URI.create(input), conf);
		FSDataInputStream inputStream = fs.open(new Path(input));
		BufferedReader buffer = new BufferedReader(new InputStreamReader(inputStream));
		String strLine = "";

		ArrayList<String> trueAndPredict = new ArrayList<>();// 直接保存真实类别+预测类别
		Set<String> trueClassNameSet = new TreeSet<>();// 保存真实类别
		while ((strLine = buffer.readLine()) != null) {
			String[] temp = strLine.split("\t");
			trueAndPredict.add(temp[0] + "\t" + temp[2]);// 直接使用真实类别+预测类别即可
			trueClassNameSet.add(temp[0]);
		}

		int trueClassNumber = trueClassNameSet.size();// 真实类别的个数
		String[] strArray = new String[trueClassNumber]; // 定义长度为trueClassNumber的字符串数组
		String[] trueClassName = (String[]) trueClassNameSet.toArray(strArray); // 将集合转换为字符串数组形式
		// 定义四个数组计算，数组大小为trueClassNumber
		int TP[] = new int[trueClassNumber];
		int FP[] = new int[trueClassNumber];
		int FN[] = new int[trueClassNumber];
		int TN[] = new int[trueClassNumber];
		for (int j = 0; j < trueAndPredict.size(); j++) {// 遍历数组
			for (int i = 0; i < trueClassName.length; i++) {// 遍历真实类别分别计算每个真实类别的tn，fn
				String[] temp = trueAndPredict.get(j).split("\t");
				if (trueClassName[i].equals(temp[0])) { // 统计属于country[0]的tp fn
					if (temp[1].equals(temp[0])) {
						TP[i]++;
					} else {
						FN[i]++;
					}
				} else {
					if (trueClassName[i].equals(temp[1])) {
						FP[i]++;
					} else {
						TN[i]++;
					}
				}
			}
		}

		// 保存评估结果到文件中，
		String output = conf.get("resultOut");
		Path outputPath = new Path(output);
		FileSystem outFs = outputPath.getFileSystem(conf);
		FSDataOutputStream outputStream = outFs.create(outputPath);
		String ctx = "";
		double P[] = new double[trueClassNumber];
		double R[] = new double[trueClassNumber];
		double F1[] = new double[trueClassNumber];
		for (int i = 0; i < trueClassNumber; i++) {
			P[i] = (double) TP[i] / (TP[i] + FP[i]);
			R[i] = (double) TP[i] / (TP[i] + FN[i]);
			F1[i] = (double) 2 * P[i] * R[i] / (P[i] + R[i]);
			ctx += trueClassName[i] + "\tP=" + P[i] + "\tR=" + R[i] + "\tF1=" + F1[i] + "\n";
			System.out.println(trueClassName[i] + ":P=" + P[i] + "\tR=" + R[i] + "\tF1=" + F1[i]);
		}
		outputStream.writeBytes(ctx);
		outputStream.close();
	}
}
