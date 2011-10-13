/**
 * @author Mark Smith (mjs66@waikato.ac.nz)
 */

package moa.streams.filters;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Hashtable;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Random;

import moa.classifiers.Classifier;
import moa.core.InstancesHeader;
import moa.core.ObjectRepository;
import moa.options.ClassOption;
import moa.options.FloatOption;
import moa.options.IntOption;
import moa.tasks.TaskMonitor;
import weka.core.Instance;
import weka.core.Instances;

import moa.streams.InstanceStream;

public class LocalGlobalFiltering extends AbstractStreamFilter {
		
	@Override
	public String getPurposeString() {
		return "Accurately identify and remove mislabeled data for more accurate models.";
	}

	/*
	 * The options after this are just a guess to the default value. Perhaps
	 * some quick research could be done into good default values?
	 */
	public ClassOption streamOption = new ClassOption("stream", 's',
			"Stream to filter.", InstanceStream.class,
			"generators.RandomTreeGenerator");
	
	// I know that there is a learner setting, but not sure how to grab it. This will do for now.
	public ClassOption learnerOption = new ClassOption("classifier", 'c',
			"Classifier used.", Classifier.class,
			"moa.classifiers.NaiveBayes");
	
	public IntOption chunkSizeOption = new IntOption("chunkSize", 'N',
			"Set the size of each chunk.", 25);

	public FloatOption removalAlphaOption = new FloatOption("removalAlpha",
			'a', "The percentage of instances to remove from the stream.", 0.1,
			0.0, 1.0);

	public IntOption nClassifiersGlobalFilteringOption = new IntOption(
			"nClassifiersGlobal", 'k',
			"Number of classifiers involved in Global Filtering", 20);

	public IntOption nFoldsLocalFilterOption = new IntOption(
			"nFoldsLocalFilter", 'n', "Number of folds for local filtering.", 5);

	public IntOption nClassifiersLocalOption = new IntOption(
			"nHeterogeneousClassifiers", 'g',
			"Number of heterogeoneous classifiers built for local filtering.",
			4, 1, 4);

	private Instances currentChunk;
	protected Classifier classifier;
	protected Classifier[] ensemble;

	private Queue<Instance> instanceReturn = new LinkedList<Instance>();

	protected int chunkSize;
	protected float removalAlpha;
	protected int nGlobalClassifiers;
	protected int nLocalFolds;
	protected int nLocalClassifiers;
	
	protected InstanceStream instanceStream;
	
	@Override
    public InstancesHeader getHeader() {
        return null;//this.inputStream.getHeader();
    }
	

	@Override
	public long estimatedRemainingInstances() {
		return this.instanceStream.estimatedRemainingInstances() + instanceReturn.size();
	}
	
	@Override
	public boolean hasMoreInstances() {
		return this.instanceStream.hasMoreInstances() || !instanceReturn.isEmpty();
	}

	@Override
	public void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {
		this.chunkSize = chunkSizeOption.getValue();
		this.removalAlpha = (float) removalAlphaOption.getValue(); 
		this.nGlobalClassifiers = nClassifiersGlobalFilteringOption.getValue();
		this.nLocalFolds = nFoldsLocalFilterOption.getValue();
		this.nLocalClassifiers = nClassifiersLocalOption.getValue();

		this.classifier = (Classifier) getPreparedClassOption(this.learnerOption);
		this.instanceStream = (InstanceStream) getPreparedClassOption(this.streamOption);
		
		//super.prepareForUseImpl(monitor, repository);
	}

	// GIT TEST

	@Override
	public Instance nextInstance() {
		// 1. Collect data from S and form a chunk Si with N instances
		
		this.initVariables();
		
		if (currentChunk.size() < chunkSize) {
			currentChunk.add(instanceStream.nextInstance());
		}
		
		if (this.currentChunk.size() == chunkSize)
			processChunk();
		
		if (instanceReturn.isEmpty())
			return null;
		else 
			return instanceReturn.poll();
	}
	
	private void initVariables() {
		if (this.currentChunk == null) {
			this.currentChunk = new Instances(instanceStream.getHeader());
		}
	}

	private void processChunk() {
		System.out.println("Processing chunk.");
		// 2. Build classifier Ci from Si
		// TODO Not sure if this is right...
		for (int i = 0; i < chunkSize; i++) {
			this.classifier.trainOnInstance(currentChunk.get(i));
		}
		
		// 3. Produce local ranking list.
		produceLocalRankingList();
		
		
		
		this.currentChunk = null;
		this.classifier = (Classifier) getPreparedClassOption(this.learnerOption);
		this.classifier.resetLearning();

	}
	
	private void produceLocalRankingList() {
		// Each datachunk is split into n-folds F1, F2, .., Fn
		Instances copy = new Instances(currentChunk);
		Instances[] folds = new Instances[this.nLocalFolds];
		
		int n = this.currentChunk.size() % this.nLocalFolds;
		int fl = this.currentChunk.size() / this.nLocalFolds;
		for (int f = 0; f < folds.length; f++) {
			if (folds[f] == null) {
				folds[f] = new Instances(instanceStream.getHeader());
			}
			
			// put fl instances in to start with
			for (int x = 0; x < fl; x++)
				folds[f].add(copy.remove(0));
			
			// and put another one in if the folds will be uneven
			if (n != 0 && f < n)
				folds[f].add(copy.remove(0));
		}
		
		// For each fold Fj, Γ classifiers are trained from Fj's complement set where
		// each of the Γ classifiers is trained by using different types of learners.	
		Classifier[][] foldClassifiers = new Classifier[folds.length][this.nLocalClassifiers];
		for (int j = 0; j < folds.length; j++) {
			foldClassifiers[j] = trainLocalClassifier(j, folds);
			
			int foldSize = folds[j].size();
		
			// For each instance x in Fj, the Γ classifiers are applied to calculate a score
			double[] instanceScores = new double[foldSize];
			for (int inst = 0; inst < foldSize; inst++) {
				instanceScores[inst] = getInstanceScores(folds[j].get(inst), foldClassifiers[j]);
			}			
		}
	}

	private double getInstanceScores(Instance inst, Classifier[] classifiers) {
		int[] classFrequencies = new int[inst.numClasses()];
		Arrays.fill(classFrequencies, 0);
		
		
		double[][] classPredictions = new double[classifiers.length][];
		for (int c = 0; c < classifiers.length; c++) { 
			classPredictions[c] = classifiers[c].getVotesForInstance(inst);
			// This comes in the form [6.0, 14.0], I'm assuming that the higher number is the class predicted.
		}
		
		// Iterate over each prediction, find the highest class probability and
		//bung that into the ArrayList, adding a number if one is already there.
		for (int c = 0; c < classPredictions.length; c++) {
			
			
			int highestClass = Integer.MIN_VALUE;
			double highestSoFar = Double.MIN_VALUE;			
			for (int p = 0; p < classPredictions[c].length; p++) {
				if (classPredictions[c][p] > highestSoFar) {
					highestSoFar = classPredictions[c][p];
					highestClass = p;
				}
			}			
			classFrequencies[highestClass] += 1;
		}
		
		// Now we have an array with the prediction sums, find the highest again to find
		// c: the class mostly agreed by all gamma classifiers.
		int c = 0;
		int highestFrequency = 0;
		for (int i = 0; i < classFrequencies.length; i++) {
			if (classFrequencies[i] > highestFrequency) {
				c = i;
				highestFrequency = classFrequencies[i];
			}
		}
		
		System.out.println("Highest class = " + c);
		return 0;
				
	}


	private Classifier[] trainLocalClassifier(int currentFold, Instances[] allFolds) {
		//Γ classifiers are trained from Fj's complement set where
		// each of the Γ classifiers is trained by using different types of learners
		Classifier[] clsArray = new Classifier[this.nLocalClassifiers];
		for (int g = 0; g < this.nLocalClassifiers; g++) {
			Classifier cls = getClassifier(g);
			
			for (int cs = 0; cs < allFolds.length; cs++) // Train from Fj's complement set
				if (cs != currentFold)
					trainOnInstances(cls, allFolds[cs]);
			
			clsArray[g] = cls;
		}		
		return clsArray;
	}


	private void trainOnInstances(Classifier cls, Instances instances) {
		for (Instance inst : instances) {
			cls.prepareForUse();
			cls.trainOnInstance(inst);
		}
	}


	/**
	 * I'm not sure if this is the best way to get different types of learners, opinion?
	 * @return
	 * @throws Exception 
	 */
	private Classifier getClassifier(int n) {
		//Random random = new Random();
		// TODO: implement more, if this works ok.
		//int n = random.nextInt(5);
		
		if (n > 4) {
			System.err.println("Breaking, incorrent integer given. Developer is stupid.");
			System.exit(-1);
		}
		
		// Can't use OzaBoost(), it generates nothing for problems with more than two classes
		// HoeffdingTree, MajorityClass and DecisionStump seem to give the same answers...
		switch(n) {
			case 0: return new moa.classifiers.NaiveBayes();
			case 1: return new moa.classifiers.HoeffdingTree();
			case 2: return new moa.classifiers.AdaHoeffdingOptionTree();
			case 3: return new moa.classifiers.HoeffdingAdaptiveTree();
		}
		
		return null;
	}
		

	@Override
	public void getDescription(StringBuilder arg0, int arg1) {
		System.out.println("GET DESCRIPTION OF SOMETHING");
	}

	@Override
	protected void restartImpl() {
		// TODO Auto-generated method stub
	}
	
	

}
