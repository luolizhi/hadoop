package org.lukey.hadoop.bayes.trainning;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.URI;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

/*
 * 测试集用了单个类别，分别是USA 50个，INDIA 20个， ARG 15个。分别计算其tp fp  fn tn
 */
public class Evaluation {
	static String[] country = { "USA", "INDIA", "ARG" };//选取三个类别
	static int TP[] = { 0, 0, 0 };
	static int FP[] = { 0, 0, 0 };
	static int FN[] = { 0, 0, 0 };
	static int TN[] = { 0, 0, 0 };

	public static void run(Configuration conf) throws Exception {
		String input = conf.get("result");
		
		//读取评估类别，计算P R F1
		FileSystem fs = FileSystem.get(URI.create(input), conf);
		FSDataInputStream inputStream = fs.open(new Path(input));
		BufferedReader buffer = new BufferedReader(new InputStreamReader(inputStream));
		String strLine = "";

		while ((strLine = buffer.readLine()) != null) {
			for (int i = 0; i < country.length; i++) {
				String[] temp = strLine.split("\t");
				if (country[i].equals(temp[0])) { // 统计属于country[0]的tp fn
					if (temp[2].equals(temp[0])) {
						TP[i]++;
					} else {
						FN[i]++;
					}
				} else {
					if (country[i].equals(temp[2])) {
						FP[i]++;
					} else {
						TN[i]++;
					}
				}
			}
		}

		//保存评估结果到文件中，
		String output = conf.get("resultOut");
		Path outputPath = new Path(output);
		FileSystem outFs = outputPath.getFileSystem(conf);
		FSDataOutputStream outputStream = outFs.create(outputPath);
		String ctx ="";
		double P[] = { 0.0, 0.0, 0.0 };
		double R[] = { 0.0, 0.0, 0.0 };
		double F1[] = { 0.0, 0.0, 0.0 };
		for (int i = 0; i < 3; i++) {
			P[i] = (double) TP[i] / (TP[i] + FP[i]);
			R[i] = (double) TP[i] / (TP[i] + FN[i]);
			F1[i] = (double) 2 * P[i] * R[i] / (P[i] + R[i]);
			ctx += country[i] + "\tP=" + P[i] +"\tR=" + R[i] + "\tF1=" + F1[i] + "\n";
			System.out.println(country[i] + "\tP=" + P[i] +"\tR=" + R[i] + "\tF1=" + F1[i]);
		}
		outputStream.writeBytes(ctx);
		outputStream.close();
	}
}
