/**
 * @author Mark Smith (mjs66@waikato.ac.nz)
 */

package moa.streams.filters;

import moa.core.InstancesHeader;
import moa.core.ObjectRepository;
import moa.options.FloatOption;
import moa.options.IntOption;
import moa.tasks.TaskMonitor;
import weka.core.Instance;
import weka.core.Instances;

public class LocalGlobalFiltering extends AbstractStreamFilter {

	@Override
	public InstancesHeader getHeader() {
		return new InstancesHeader(this.instances);
	}
	
	@Override
	public String getPurposeString() {
		return "Accurately identify and remove mislabeled data for more accurate models.";
	}
	
	/* The options after this are just a guess to the default value. Perhaps some quick 
	 * research could be done into good default values? 
	 */	
	public IntOption chunkSizeOption = new IntOption("chunkSize", 'N', "Set the size of each chunk.", 20);
	
	public FloatOption removalAlphaOption = new FloatOption("removalAlpha", 'a', 
			"The percentage of instances to remove from the stream.", 0.1, 0.0, 1.0);
	
	public IntOption nClassifiersGlobalFilteringOption = new IntOption("nClassifiersGlobal", 'k', 
			"Number of classifiers involved in Global Filtering", 20);
	
	public IntOption nFoldsLocalFilterOption = new IntOption("nFoldsLocalFilter", 'n', 
			"Number of folds for local filtering.", 5);
	
	public IntOption nHeterogeneousClassifiersLocalOption = new IntOption("nHeterogeneousClassifiers", 'g',
			"Number of heterogeoneous classifiers built for local filtering.", 10);

	private Instances instances;
	
	protected int chunkSize;
	protected float removalAlpha;
	protected int nGlobalClassifiers;
	protected int nLocalFolds;
	protected int nHeterogeneousClassifiers;
	
	@Override
	public void	prepareForUseImpl (TaskMonitor monitor, ObjectRepository repository) {
		this.chunkSize 					= chunkSizeOption.getValue();
		this.removalAlpha 				= (float) removalAlphaOption.getValue();	// Perhaps a bug???
		this.nGlobalClassifiers 		= nClassifiersGlobalFilteringOption.getValue();
		this.nLocalFolds 				= nFoldsLocalFilterOption.getValue();
		this.nHeterogeneousClassifiers 	= nHeterogeneousClassifiersLocalOption.getValue();
		
		super.prepareForUse(monitor, repository);
	}
	
	@Override
	public Instance nextInstance() {
		
		if (instances == null) {
			this.instances = new Instances(instances, chunkSize);
		}
		
		if (this.instances.size() < chunkSize) {
			this.instances.add(instance)
		}
		
		return null;
	}

	private void processChunk(Instances instances2) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void getDescription(StringBuilder arg0, int arg1) {
		// TODO Auto-generated method stub

	}

	@Override
	protected void restartImpl() {
		// TODO Auto-generated method stub
	}

}
