package org.lukey.hadoop.bayes.trainning;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Counter;
import org.apache.hadoop.mapreduce.Counters;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

/**
 * 
 * ������Ϊ��������ʵ���+�ļ���+������������Ԥ����� ������ʵ����P��R��F1 ����ĳһ����ʵ���AAA�������һ����ʵ���ΪAAA
 * ���������������ʵ��ͬ��tp��1�� ���������������ʵ���ͬ��fn��1�� �����һ�в���AAA�����ҵ�������AAA��fp��1������tn��1.
 * 
 */
public class Predict {
	static enum counter {// �ֱ��ʾ�����ı��ĵ������������//δ�ܺú�ʹ��
		words_in_file, class_counte
	}
	static Map<String, Double> priorMap = new HashMap<>();// ����������� ALB 0.375
	static Map<String, Map<String, Double>> conditionMap = new HashMap<>();//��������
	static Map<String, Double> notFoundMap = new HashMap<>();//�����û�е��ʵĸ���
	// ���������������ҪԤ����ļ���ѭ�����㲻ͬ���ĸ��ʣ���ʶֵ��Ϊ�������
	// ÿ��ȡvalue�ۼӣ���Ҫ��Math.log����
	static Map<String, Double> predictMap = new HashMap<String, Double>();//Ԥ������͸���
	public static int run(Configuration conf) throws Exception {
		conf.set("mapred.job.tracker", "192.168.190.128:9001");

		Job job = new Job(conf, "predict");
		job.setJarByClass(Predict.class);

		job.setInputFormatClass(PredictInputFormat.class);

		job.setMapperClass(PredictMapper.class);
		job.setReducerClass(PredictReducer.class);

		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);

		String testInput = conf.get("testInput");

		List<Path> paths = MyUtils.getSecondDir(conf, testInput);
		for (Path path : paths) {
			PredictInputFormat.addInputPath(job, path);
		}

		String testOutput = conf.get("testOutput");
		FileOutputFormat.setOutputPath(job, new Path(testOutput));

		int exitCode = job.waitForCompletion(true) ? 0 : 1;
		// ���ü�������ֻ���������Բ鿴
		Counters counters = job.getCounters();
		Counter c1 = counters.findCounter(counter.class_counte);
		System.out.println(c1.getDisplayName() + " : " + c1.getValue());// ��������ļ��������Ͳ⼯�����Եĳ˻�
		return exitCode;
	}

	public static class PredictMapper extends Mapper<Text, Text, Text, Text> {

		protected void setup(Mapper<Text, Text, Text, Text>.Context context)
				throws IOException, InterruptedException {
			Configuration conf = context.getConfiguration();
			
			String priorPath = conf.get("priorProbality");// ��ȡ�������
			priorMap = MyUtils.getProbability(conf, priorPath);//���÷���

			String notFoundPath = conf.get("notFoundPath");// ��ȡÿ�����û���ҵ����ʵĸ���
			notFoundMap = MyUtils.getProbability(conf, notFoundPath);//���÷���

			String conditionPath = conf.get("conditionPath");// ��ȡ��������
			Path condPath = new Path(conditionPath);
			FileSystem fs = condPath.getFileSystem(conf);
			FileStatus[] stats = fs.listStatus(condPath);
			for (FileStatus stat : stats) {
				if (!stat.isDir()) {
					Path filePath = stat.getPath();
					String fileName = filePath.getName();
					String[] temp = fileName.split("-");
					if (temp.length == 3) {
						String className = temp[0];// �õ������			
						Map<String, Double> oneMap = new HashMap<>();// �����ļ�·����ȡ�ļ��������ݱ��浽map
						oneMap = MyUtils.getProbability(conf, filePath.toString());
						conditionMap.put(className, oneMap);
					}
				}
			}
		}

		protected void map(Text key, Text value, Mapper<Text, Text, Text, Text>.Context context)
				throws IOException, InterruptedException {
			// ����preditMap�����Ӧ������иõ���value�ͳ�����ʣ����ڳ���notFoundMap����ĸ���
			/* Map<String, Map<String, Double>> conditionMap */
			for (String className : priorMap.keySet()) {// �������п��ܵ����
				context.getCounter(counter.class_counte).increment(1);// ÿ��map����ͳ�����е����
				// �þ�̬����������ĵ������ĸ����ĸ���
				double p = conditionalProbabilityForClass(value.toString(), className);
				Text prob = new Text(className + "\t" + p);
				context.write(key, prob);//key�����+�ļ���
			}
		}

		public static double conditionalProbabilityForClass(String content, String className) {
			 // ����ĳ���ı�����ĳһ���ĸ���// className����ÿ�����ʵĸ���
			Map<String, Double> condMap = conditionMap.get(className);// class�����г��ֵ��ʵĸ���
			double notFindProbability = notFoundMap.get(className);// class��û�г��ֵ��ʵĸ���
			double priorProbability = priorMap.get(className);// class���������
			double pClass = Math.log(priorProbability);// ʹ��ʼ���ʵ���������ʣ�log����
			// ���������ı��е����е��ʣ���ÿ�����ʵĸ��ʼ���log�����
			StringTokenizer itr = new StringTokenizer(content.toString());
			while (itr.hasMoreTokens()) {
				String word = itr.nextToken();
				if (condMap.containsKey(word)) {// ������к��иõ��ʵ���������
					pClass += Math.log(condMap.get(word));
				} else {// ����𲻺��õ��ʵĸ���
					pClass += Math.log(notFindProbability);
				}
			}
			return pClass;// ��������������������
		}

	}

	public static class PredictReducer extends Reducer<Text, Text, Text, Text> {

		@Override
		protected void reduce(Text key, Iterable<Text> values, Reducer<Text, Text, Text, Text>.Context context)
				throws IOException, InterruptedException {
			// find the max probability
			double maxP = -100000000.0;
			String maxClass = "";
			for (Text value : values) {// �����ı������������������ĸ���
				String[] temp = value.toString().split("\t");
				double p = Double.parseDouble(temp[1]);
				if (p > maxP) {//���ʴ������ֵ
					maxP = p;//����������ֵ
					maxClass = temp[0];//�������ֵ�����
				}
			}
			context.write(key, new Text(maxClass));
		}
	}
}