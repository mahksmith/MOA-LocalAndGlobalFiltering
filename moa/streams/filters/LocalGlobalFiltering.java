/**
 * @author Mark Smith (mjs66@waikato.ac.nz)
 */

package moa.streams.filters;

import moa.core.InstancesHeader;
import moa.options.FloatOption;
import moa.options.IntOption;
import weka.core.Instance;

public class LocalGlobalFiltering extends AbstractStreamFilter {

	@Override
	public InstancesHeader getHeader() {
		// TODO Auto-generated method stub
		return null;
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
			"The percentage of instances to remove from the stream.", 0.05, 0.0, 1.0);
	
	public IntOption nClassifiersGlobalFilteringOption = new IntOption("nClassifiersGlobal", 'k', 
			"Number of classifiers involved in Global Filtering", 20);
	
	public IntOption nFoldsLocalFilterOption = new IntOption("nFoldsLocalFilter", 'n', 
			"Number of folds for local filtering.", 5);
	
	public IntOption nHeterogeneousClassifiersLocalOption = new IntOption("nHeterogeneousClassifiers", 'g',
			"Number of heterogeoneous classifiers built for local filtering.", 10);

	// TODO: maybe later we could add some noise to this for debugging purposes.
	
	@Override
	public Instance nextInstance() {
		// TODO Auto-generated method stub
		return null;
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
