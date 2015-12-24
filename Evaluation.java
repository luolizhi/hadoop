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
 * ���Լ����˵�����𣬷ֱ���USA 50����INDIA 20���� ARG 15�����ֱ������tp fp  fn tn
 */
public class Evaluation {

	public static void run(Configuration conf) throws Exception {
		String input = conf.get("result");

		// ��ȡ������𣬼���P R F1
		FileSystem fs = FileSystem.get(URI.create(input), conf);
		FSDataInputStream inputStream = fs.open(new Path(input));
		BufferedReader buffer = new BufferedReader(new InputStreamReader(inputStream));
		String strLine = "";

		ArrayList<String> trueAndPredict = new ArrayList<>();// ֱ�ӱ�����ʵ���+Ԥ�����
		Set<String> trueClassNameSet = new TreeSet<>();// ������ʵ���
		while ((strLine = buffer.readLine()) != null) {
			String[] temp = strLine.split("\t");
			trueAndPredict.add(temp[0] + "\t" + temp[2]);// ֱ��ʹ����ʵ���+Ԥ����𼴿�
			trueClassNameSet.add(temp[0]);
		}

		int trueClassNumber = trueClassNameSet.size();// ��ʵ���ĸ���
		String[] strArray = new String[trueClassNumber]; // ���峤��ΪtrueClassNumber���ַ�������
		String[] trueClassName = (String[]) trueClassNameSet.toArray(strArray); // ������ת��Ϊ�ַ���������ʽ
		// �����ĸ�������㣬�����СΪtrueClassNumber
		int TP[] = new int[trueClassNumber];
		int FP[] = new int[trueClassNumber];
		int FN[] = new int[trueClassNumber];
		int TN[] = new int[trueClassNumber];
		for (int j = 0; j < trueAndPredict.size(); j++) {// ��������
			for (int i = 0; i < trueClassName.length; i++) {// ������ʵ���ֱ����ÿ����ʵ����tn��fn
				String[] temp = trueAndPredict.get(j).split("\t");
				if (trueClassName[i].equals(temp[0])) { // ͳ������country[0]��tp fn
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

		// ��������������ļ��У�
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
