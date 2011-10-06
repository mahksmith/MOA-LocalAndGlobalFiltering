/**
 * @author Mark Smith (mjs66@waikato.ac.nz)
 */

package moa.streams.filters;

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
	
	public IntOption chunkSizeOption = new IntOption("chunkSize", 'N',
			"Set the size of each chunk.", 20);

	public FloatOption removalAlphaOption = new FloatOption("removalAlpha",
			'a', "The percentage of instances to remove from the stream.", 0.1,
			0.0, 1.0);

	public IntOption nClassifiersGlobalFilteringOption = new IntOption(
			"nClassifiersGlobal", 'k',
			"Number of classifiers involved in Global Filtering", 20);

	public IntOption nFoldsLocalFilterOption = new IntOption(
			"nFoldsLocalFilter", 'n', "Number of folds for local filtering.", 5);

	public IntOption nHeterogeneousClassifiersLocalOption = new IntOption(
			"nHeterogeneousClassifiers", 'g',
			"Number of heterogeoneous classifiers built for local filtering.",
			10);

	private Instances instances;

	protected int chunkSize;
	protected float removalAlpha;
	protected int nGlobalClassifiers;
	protected int nLocalFolds;
	protected int nHeterogeneousClassifiers;
	
	protected InstanceStream instanceStream;
	
	@Override
    public InstancesHeader getHeader() {
        return null;//this.inputStream.getHeader();
    }
	

	@Override
	public long estimatedRemainingInstances() {
		/* TODO: this is only a temporary fix */
		return this.instanceStream.estimatedRemainingInstances();
	}
	
	@Override
	public boolean hasMoreInstances() {
		/* TODO: this is only a temporary fix, final version will need to be
		 * this.instanceStream.hasMoreInstances() || ![instanceoutputqueue].isEmpty()
		 */
		return this.instanceStream.hasMoreInstances();
	}

	@Override
	public void prepareForUseImpl(TaskMonitor monitor,
			ObjectRepository repository) {
		this.chunkSize = chunkSizeOption.getValue();
		this.removalAlpha = (float) removalAlphaOption.getValue(); // Perhaps a
																	// bug???
		this.nGlobalClassifiers = nClassifiersGlobalFilteringOption.getValue();
		this.nLocalFolds = nFoldsLocalFilterOption.getValue();
		this.nHeterogeneousClassifiers = nHeterogeneousClassifiersLocalOption
				.getValue();

		// Shamelessly stolen from moa.streams.FilteredStream
		this.instanceStream = (InstanceStream) getPreparedClassOption(this.streamOption);
	}

	// GIT TEST

	@Override
	public Instance nextInstance() {
		// TODO FINISH THIS
		//return this.instanceStream.nextInstance();
		
		// 1. Collect data from S and form a chunk Si with N instances
		// TODO Instances(java.lang.String name, java.util.ArrayList<Attribute> attInfo, int capacity)
        //Creates an empty set of instances.
		return this.instanceStream.nextInstance();
	}

	private void processChunk(Instances instances) {

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
