package edu.upenn.cis.swell.IO;

/**
 * ver: 1.0
 * @author paramveer dhillon.
 *
 * last modified: 09/04/13
 * please send bug reports and suggestions to: dhillon@cis.upenn.edu
 */

import java.io.File;
import java.io.Serializable;
import java.util.ArrayList;

public final class Options implements Serializable {
	
 static final long serialVersionUID = 42L;
 public String unlabDataTrainfile = null;
 //public String ccaAlgorithm = null;
 public String algorithm = null;
 public String serializeRep = null;
 public String serializeRun = null;
 public String typeofDecomp= null;
 public String serializeCorpus = null;
 public String trainfile = null;
 public String testfile = null;
 public String contextSpecificEmbed=null;
 public String contextOblEmbed=null;
 public String contextOblEmbedContext=null;
 public String eigenEmbedFile=null;
 public boolean train = false;
 public boolean trainUnlab = false;
 public boolean eval = false;
 public boolean test = false;
 public boolean lowercase = false;
 public boolean bagofWordsSVD = false;
 public boolean scaleBySingVals=false;
 public boolean induceEmbeds=false;
 public boolean ngrams = false;
 public boolean writeContextMatrix=false;
 public int numLabels = 1;
 public boolean normalize = false;
 public boolean parallel = false;
 public String eigendictName = "eigenWord.dict";
 public String docSeparator = "DOCSTART-X-0";
 public String lSVecName=null;
 public String rSVecName=null;
 public boolean pruneStopSymbols=false;
 public ArrayList<Double> smoothArray=new  ArrayList<Double>();
 public int vocabSize = 30000;
 public int hiddenStateSize=50;
 public int contextSizeOneSide=2;
 public boolean randomBaseline=false;
 public int numIters = 2;
 public int numGrams=2;
 public String outfile = "out.txt";
 public int totalContextSizeLplusR=1;
 public boolean normalizePCA=false;
 public boolean depbigram=false;
 public boolean diagOnlyInverse=false;
 public boolean contextSensitive=false;
 public boolean descendingVocab=false;
 public String outputDir="Output_Files/";
 public String embedToInduce=null;
 public String eigenWordCCAFile=null;
 public boolean kdimDecomp=false,logTrans=false,sqRootTrans=false;
 public boolean sqRootNorm=false;
 public int n=0,p=0;
 
 
 
 public Options (String[] args) {

	for(int i = 0; i < args.length; i++) {
	    String[] pair = args[i].split(":");

	    if (pair[0].equals("algorithm")) {
	    	algorithm = pair[1];
	    	typeofDecomp = pair[2];
		}
	    
	    
	 //Default file names; if you need to change them uncomment later in this file and make these command line parameters.
	   
	    if(pair[0].equals("output-dir-prefix")){
	    	outputDir=pair[1]+"/";
	    	
	    }
	    
	    
	    if(pair[0].equals("eigenCCA-dict-file")){
	    	eigenWordCCAFile=pair[1];
	    	n=Integer.parseInt(pair[2]);
	    	p=Integer.parseInt(pair[3]);
	    }
	    if(pair[0].equals("eigen-embed-file")){
	    	eigenEmbedFile=pair[1];
	    	n=Integer.parseInt(pair[2]);
	    	p=Integer.parseInt(pair[3]);
	    }
	    
	    
	    
	    
	    if(pair[0].equals("embed-to-induce")){
	    	embedToInduce=pair[1];
	    	
	    }
	    
	    contextSpecificEmbed = outputDir+"contextSpecificEmbed."+algorithm;
	    
	    
	    
	    if(pair[0].equals("context-specific-name")){
	    	contextSpecificEmbed=outputDir+ pair[1];
	    }
	    
	  	    
	    File folder = new File(outputDir);
    	folder.mkdirs();
	    
	    serializeRun = outputDir+"run."+algorithm;
		
	    serializeRep = outputDir+"rep."+algorithm;
	 
	    serializeCorpus = outputDir+"corpus."+algorithm;
	    	
	    contextOblEmbed = outputDir+"contextObliviousEmbed."+algorithm;
	    
	    contextOblEmbedContext = outputDir+"contextObliviousEmbedContext."+algorithm;
	
		eigendictName = outputDir+"eigenDict."+algorithm;
		
		lSVecName = outputDir+"PhiLSingVect."+algorithm;
		
		rSVecName = outputDir+"PhiLSingVect."+algorithm;
		
		if (pair[0].equals("writeContextMatrix"))
		{
			writeContextMatrix=true;
		}
		
		if (pair[0].equals("descendingVocab"))
		{
			descendingVocab=true;
		}
		
		
		
		if (pair[0].equals("context-sensitive"))
		{
			contextSensitive=true;
		}
		
		if (pair[0].equals("parallel"))
		{
			parallel=true;
		}
		
		if (pair[0].equals("kdim-decomp"))
		{
			kdimDecomp=true;
		}
		
		if (pair[0].equals("sqroot-norm"))
		{
			sqRootNorm=true;
		}
		if (pair[0].equals("log-trans"))
		{
			logTrans=true;
		}
		if (pair[0].equals("sqrt-trans"))
		{
			sqRootTrans=true;
		}
	
	    
	    if (pair[0].equals("diagOnlyInverse")) {
	    	diagOnlyInverse = true;
	    }
	    
	    if (pair[0].equals("train")) {
			train = true;
		    }
	    
	    if (pair[0].equals("no-random")) {
			 randomBaseline= false;
		    }
	    
	    if (pair[0].equals("eval")) {
		eval = true;
	    }
	    if (pair[0].equals("lowercase")) {
			lowercase = true;
		  }
	    if (pair[0].equals("dep-bigram")) {
			depbigram = true;
		  }
	    
	    if (pair[0].equals("num-grams")) {
			numGrams = Integer.parseInt(pair[1]);
		  }
	    if (pair[0].equals("normalize")) {
			normalize = true;
		  }
	    
	    if (pair[0].equals("normalizePCA")) {
	    	normalizePCA = true;
		  }
	    if (pair[0].equals("test")) {
		test = true;
	    }
	    if (pair[0].equals("bagofWordsSVD")) {
	    	bagofWordsSVD = true;
		    }
	    if (pair[0].equals("ngrams")) {
	    	ngrams = true;
		    }
	    if (pair[0].equals("scaleDictBySingVals")) {
	    	scaleBySingVals = true;
		    }
	    
	    if (pair[0].equals("induce-embeds")) {
	    	induceEmbeds = true;
		    }
	    
	    if (pair[0].equals("contextSizeEachSide")) {
	    	contextSizeOneSide = Integer.parseInt(pair[1]);
		    }
	    
	    if (pair[0].equals("totalContextSizeLplusR")) {
	    	totalContextSizeLplusR = Integer.parseInt(pair[1]);
		    }
	    
	    if (pair[0].equals("smooths")) {
	    	String[] smooths=pair[1].split(",");
	    	for(int j=0;j<smooths.length;j++)
	    		smoothArray.add(Double.parseDouble(smooths[j]));
		    }
	    
	    if (pair[0].equals("iters")) {
		numIters = Integer.parseInt(pair[1]);
	    }
	    
	    
	    if (pair[0].equals("num-labels")) {
			numLabels = Integer.parseInt(pair[1]);
	
	  }
	  
	    /*
	    if (pair[0].equals("serialize-run-file")) {
	    	serializeRun = "Output_Files/run"+algorithm +".ser";
		    }
	    if (pair[0].equals("serialize-rep.-file")) {
	    	serializeRep = "Output_Files/rep"+algorithm +".ser";
	    }
	    if (pair[0].equals("serialize-corpus-file")) {
	    	serializeCorpus = "Output_Files/corpus"+algorithm +".ser";
		    }
	    if (pair[0].equals("context-specific-embed-name")) {
	    	contextSpecificEmbed = "Output_Files/contextSpecificEmbed"+algorithm +".ser";
		    }
	    if (pair[0].equals("context-obl-embed-name")) {
	    	contextOblEmbed = "Output_Files/contextObliviousEmbed"+algorithm +".ser";
		    }
	    
	    if (pair[0].equals("eigen-dict-name")) {
			eigendictName = "Output_Files/eigenDict"+algorithm;
		    }
		    if (pair[0].equals("left-singular-vec-name")) {
				lSVecName = "Output_Files/PhiLSingVect"+algorithm;
			    }
		    if (pair[0].equals("right-singular-vec-name")) {
				rSVecName = "Output_Files/PhiLSingVect"+algorithm;
			    }
		    
	    */
	    
	    if (pair[0].equals("output-file")) {
		outfile = pair[1];
	    }
	    
	    if (pair[0].equals("hidden-state")) {
			hiddenStateSize = Integer.parseInt(pair[1]);
		    }
	    
	    if (pair[0].equals("doc-separator")) {
	    	docSeparator = pair[1];
		 }
	    if (pair[0].equals("train-file")) {
		trainfile = pair[1];
	    }
	    if (pair[0].equals("prune-stop-symbols")) {
	    	pruneStopSymbols = true;
		}
	    
	   
	    if (pair[0].equals("unlab-train-file")) {
	    	unlabDataTrainfile = pair[1];
		}
	    
	  //  if (pair[0].equals("cca-algorithm")) {
	  //  	ccaAlgorithm = pair[1];
		//}
	    
	  
	    if (pair[0].equals("unlab-train")) {
	    	trainUnlab = true;
		}
	    
	    if (pair[0].equals("vocab-size")) {
			vocabSize = Integer.parseInt(pair[1]);
		}
	    if (pair[0].equals("test-file")) {
		testfile = pair[1];
	    }
	   
	  
	   	}


	 }

  
 public String toString () {
	StringBuilder sb = new StringBuilder();
	sb.append("FLAGS [");
	sb.append("train-file: " + trainfile);
	sb.append(" | ");
	sb.append("test-file: " + testfile);
	sb.append(" | ");
	sb.append("output-file: " + outfile);
	sb.append(" | ");
	sb.append("ngrams: " + ngrams);
	sb.append(" | ");
	sb.append("scaleDictBySingVals: " + scaleBySingVals);
	sb.append(" | ");
	sb.append("numLabels: " + numLabels);
	sb.append(" | ");
	sb.append("serialize-corpus-file: " + serializeCorpus);
	sb.append(" | ");
	sb.append("context-specific-embed-name: " + contextSpecificEmbed);
	sb.append(" | ");
	sb.append("context-obl-embed-name: " + contextOblEmbed);
	sb.append(" | ");
	sb.append("contextSizeEachSide: " + contextSizeOneSide);
	sb.append(" | ");
	sb.append("left-singular-vec-name: " + lSVecName);
	sb.append(" | ");
	sb.append("right-singular-vec-name: " + rSVecName);
	sb.append(" | ");
	sb.append("serialize-run-file: " + serializeRun);
	sb.append(" | ");
	sb.append("serialize-rep.-file: " + serializeRep);
	sb.append(" | ");
	sb.append("unlab-train-file: " + unlabDataTrainfile);
	sb.append(" | ");
	sb.append("unlab-train: " + trainUnlab);
	sb.append(" | ");
	sb.append("bagofWordsSVD: " + bagofWordsSVD);
	sb.append(" | ");
	sb.append("hidden-state: " + hiddenStateSize);
	sb.append(" | ");
	sb.append("prune-stop-symbol: " + pruneStopSymbols);
	sb.append(" | ");
	sb.append("smooths: " + smoothArray);
	sb.append(" | ");
	sb.append("doc-separator: " + docSeparator);
	sb.append(" | ");
	sb.append("algorithm: " + algorithm);
	sb.append(" | ");
	sb.append("voab-size: " + vocabSize);
	sb.append(" | ");
	sb.append("eigen-dict-name: " + eigendictName);
	sb.append(" | ");
	sb.append("lowercase: " + lowercase);
	sb.append(" | ");
	sb.append("normalize: " + normalize);
	sb.append(" | ");
	sb.append("train: " + train);
	sb.append(" | ");
	sb.append("test: " + test);
	sb.append(" | ");
	sb.append("eval: " + eval);
	sb.append(" | ");
	sb.append("iters: " + numIters);
	sb.append(" | ");
	sb.append("]\n");
	return sb.toString();
 }
}

