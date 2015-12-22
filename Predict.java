package org.lukey.hadoop.bayes.trainning;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Counter;
import org.apache.hadoop.mapreduce.Counters;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;


/**
 * 
 * ������Ϊ��������ʵ���+�ļ���+������������Ԥ�����
 * ������ʵ����P��R��F1
 * ����ĳһ����ʵ���AAA�������һ����ʵ���ΪAAA
 * ���������������ʵ��ͬ��tp��1��
 * ���������������ʵ���ͬ��fn��1��
 * �����һ�в���AAA�����ҵ�������AAA��fp��1������tn��1.
 * 
 */
public class Predict {
	
//	 private static final Log LOG = LogFactory.getLog(Predict.class);

	static enum counter {// �ֱ��ʾ�����ı��ĵ������������//δ�ܺú�ʹ��
		words_in_file, class_counte
	}
	
	static Map<String, Double> priorMap = new HashMap<>();//����������� ALB 0.375

	// ÿ��������浥�ʵĸ��� abassi 3.476341324572953E-5 ��������ļ�������Ҫ���⴦��
	// �����뷨�ǽ��ļ���������������Ϊmap��key�������������Ϊmap��value������ ���ʣ�
	static Map<String, Map<String, Double>> conditionMap = new HashMap<>();

	// ÿ���������û���ҵ��ĵ��ʵĸ���ALB 4.062444649191655E-7
	static Map<String, Double> notFoundMap = new HashMap<>();

	// ���������������ҪԤ����ļ���ѭ�����㲻ͬ���ĸ��ʣ���ʶֵ������Ϊ�������
	// ������� ALB 0.375��key���ļ���+��������
	// ÿ��ȡvalue�۳ɣ���Ҫ��log����
	static Map<String, Double> predictMap = new HashMap<String, Double>();

	public static int run(Configuration conf) throws Exception  {
		conf.set("mapred.job.tracker", "192.168.190.128:9001");
		
		Job job = new Job(conf, "predict");
		job.setJarByClass(Predict.class);

		job.setInputFormatClass(WholeFileInputFormat.class);

		job.setMapperClass(PredictMapper.class);
		job.setReducerClass(PredictReducer.class);

		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);

		String testInput = conf.get("testInput");
		
		List<Path> paths = MyUtils.getSecondDir(conf, testInput);
		for (Path path : paths) {
			WholeFileInputFormat.addInputPath(job, path);
		}

		String testOutput = conf.get("testOutput");
		FileOutputFormat.setOutputPath(job, new Path(testOutput));

		int exitCode = job.waitForCompletion(true) ? 0 : 1;

		// ���ü�����
		Counters counters = job.getCounters();
		Counter c1 = counters.findCounter(counter.class_counte);
		System.out.println(c1.getDisplayName() + " : " + c1.getValue());

		return exitCode;
	}

	// map
	public static class PredictMapper extends Mapper<LongWritable, Text, Text, Text> {

		protected void setup(Mapper<LongWritable, Text, Text, Text>.Context context)
				throws IOException, InterruptedException {
			
			Configuration conf = context.getConfiguration();
			
			String priorPath = conf.get("priorProbality");// ��ȡ�������
			priorMap = MyUtils.getProbability(conf, priorPath);
			
			String notFoundPath = conf.get("notFoundPath");// ��ȡÿ�����û���ҵ����ʵĸ���
			notFoundMap = MyUtils.getProbability(conf, notFoundPath);
						
			String conditionPath = conf.get("conditionPath");//��ȡ��������
			try {
				conditionMap = MyUtils.getConditionMap(conf, conditionPath);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
			
		protected void map(LongWritable key, Text value, Mapper<LongWritable, Text, Text, Text>.Context context)
				throws IOException, InterruptedException {

			FileSplit split = (FileSplit)context.getInputSplit();//��context��ȡsplit����ǿת��fileSplit
			Path file = split.getPath();//��ȡ�ļ�����·��
			String fileName = file.getName(); // �õ��ļ����������ʱ��д��context
			String trueClassName = file.getParent().getName() ;//�õ���ʵ���
					
			// ����preditMap�����Ӧ������иõ���value�ͳ�����ʣ����ڳ���notFoundMap����ĸ���
			/* Map<String, Map<String, Double>> conditionMap */

			for (String className : priorMap.keySet()) {//�������п��ܵ����
				context.getCounter(counter.class_counte).increment(1);//ÿ��map����ͳ�����е����
				// �þ�̬����������ĵ������ĸ����ĸ���
				double p = conditionalProbabilityForClass(value.toString(), className);
//				LOG.info(className + "------->" + p);//��һ���ı���������¼���ĸ���д��log��
//				System.out.println(className + "------->" + p);//�����log�е�stdout��
				Text prob = new Text(className + "\t" + p);
				context.write(new Text(trueClassName + "\t"+ fileName), prob);
			}
		}

		public static double conditionalProbabilityForClass(String content, String className) {
			/*
			 * ����ĳ���ı�����ĳһ���ĸ���
			 */
			// className����ÿ�����ʵĸ���
			Map<String, Double> condMap = conditionMap.get(className);//class�����г��ֵ��ʵĸ���
			double notFindProbability = notFoundMap.get(className);//class��û�г��ֵ��ʵĸ���
			double priorProbability = priorMap.get(className);//class���������
			double pClass = Math.log(priorProbability);//ʹ��ʼ���ʵ���������ʣ�log����
			//���������ı��е����е��ʣ���ÿ�����ʵĸ��ʼ���log�����
			StringTokenizer itr = new StringTokenizer(content.toString());
			while (itr.hasMoreTokens()) {
				String word = itr.nextToken();
				if (condMap.containsKey(word)) {//������к��иõ��ʵ���������
					pClass += Math.log(condMap.get(word));
				} else {//����𲻺��õ��ʵĸ���
					pClass += Math.log(notFindProbability);
				}
			}
			return pClass;//��������������������
		}

	}

	// reduce
	public static class PredictReducer extends Reducer<Text, Text, Text, Text> {

		Double maxP = 0.0;
		@Override
		protected void reduce(Text key, Iterable<Text> values, Reducer<Text, Text, Text, Text>.Context context)
				throws IOException, InterruptedException {
			// find the max probability
			double maxP = -100000000.0;
			String maxClass = "";
			for (Text value : values) {//�����ı������������������ĸ���
				String[] temp = value.toString().split("\t");
				double p = Double.parseDouble(temp[1]);
				maxClass = (p>maxP)? temp[0]: maxClass;//double�ȴ�С����ֱ�ӱȣ����Ҫ��absС��һ����С����
			}
			context.write(key, new Text(maxClass));
		}
	}

}