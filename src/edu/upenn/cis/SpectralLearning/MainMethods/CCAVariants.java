package edu.upenn.cis.SpectralLearning.MainMethods;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;

import Jama.Matrix;
import edu.upenn.cis.SpectralLearning.IO.ContextPCAWriter;
import edu.upenn.cis.SpectralLearning.IO.Options;
import edu.upenn.cis.SpectralLearning.IO.ReadDataFile;
import edu.upenn.cis.SpectralLearning.Runs.CCAVariantsRun;
import edu.upenn.cis.SpectralLearning.SpectralRepresentations.ContextPCARepresentation;

public class CCAVariants implements Serializable {
 
	static final long serialVersionUID = 42L;
	
	
	public static void main(String[] args) throws Exception{
		
		ArrayList<ArrayList<Integer>> all_Docs;
		ArrayList<Integer> docSize;
		ReadDataFile rin;
		//Corpus corpus;
		ContextPCARepresentation contextPCARep;
		HashMap<String,Integer> corpusInt=new HashMap<String,Integer>();
		HashMap<String,Integer> corpusIntOldMapping=new HashMap<String,Integer>();
		CCAVariantsRun ccaVariantRun;
		ContextPCAWriter wout;
		Object[] matrices=new Object[2];
		long numTokens;
		
		Options opt=new Options(args);
		
		if(opt.algorithm==null){
			System.out.println("WARNING: YOU NEED TO SPECIFY A VALID ALGORITHM NAME AS algorithm:");
		}
		if(opt.trainUnlab){
			System.out.println("+++Inducing CCA Embedddings from unlabeled data+++\n");
			all_Docs=new ArrayList<ArrayList<Integer>>();
			docSize=new ArrayList<Integer>();
			rin=new ReadDataFile(opt);
			
			long maxMemory = Runtime.getRuntime().maxMemory();
		    /* Maximum amount of memory the JVM will attempt to use */
		    System.out.println("Maximum memory (bytes): " + 
		        (maxMemory == Long.MAX_VALUE ? "no limit" : maxMemory));

		    /* Total memory currently in use by the JVM */
		    System.out.println("Total memory (bytes): " + 
		        Runtime.getRuntime().totalMemory());
			
			corpusInt= rin.convertAllDocsInt(0);
			rin.readAllDocs(0);
			all_Docs=rin.getAllDocs();
			docSize=rin.getDocSizes();
			numTokens=rin.getNumTokens();
			rin.serializeCorpusIntMapped();
			//corpus=new Corpus(all_Docs,docSize,opt);
			
			contextPCARep= new ContextPCARepresentation(opt, numTokens,rin,all_Docs);
			
		    /* Total memory currently in use by the JVM */
		    System.out.println("Total memory (bytes): " + 
		        Runtime.getRuntime().totalMemory());
			
			
			ccaVariantRun=new CCAVariantsRun(opt,contextPCARep);
			ccaVariantRun.serializeCCAVariantsRun();
			matrices=deserializeCCAVariantsRun(opt);

			wout=new ContextPCAWriter(opt,all_Docs,matrices,rin);
			wout.writeEigenDict();
			if(!opt.typeofDecomp.equals("TwoStepLRvsW"))
				wout.writeEigContextVectors();
			
			
			if (opt.randomBaseline){
				wout.writeEigenDictRandom();
				wout.writeEigContextVectorsRandom();
			}
			
			
		    /* Total memory currently in use by the JVM */
		    System.out.println("Total memory (bytes): " + 
		        Runtime.getRuntime().totalMemory());
			
			
			System.out.println("+++CCA Embedddings Induced+++\n");
		}
		if (opt.train){
			System.out.println("+++Generating CCA Embedddings for training data+++\n");
			all_Docs=new ArrayList<ArrayList<Integer>>();
			docSize=new ArrayList<Integer>();
			corpusIntOldMapping=deserializeCorpusIntMapped(opt);
			rin=new ReadDataFile(opt);
			rin.setCorpusIntMapped(corpusIntOldMapping);
			rin.readAllDocs(1);
			all_Docs=rin.getAllDocs();
			docSize=rin.getDocSizes();
			numTokens=rin.getNumTokens();
			//corpus=new Corpus(all_Docs,docSize,opt);
			matrices=deserializeCCAVariantsRun(opt);
			contextPCARep= new ContextPCARepresentation(opt, numTokens,rin,all_Docs);
			
			Matrix contextSpecificEmbed=contextPCARep.generateProjections((Matrix)matrices[0], 
					(Matrix)matrices[1], (Matrix)matrices[2]);
			
			Matrix contextObliviousEmbed=contextPCARep.getContextOblEmbeddings((Matrix)matrices[0]);
	
			wout=new ContextPCAWriter(opt,all_Docs,matrices,rin);
			wout.writeContextSpecificEmbed(contextSpecificEmbed);
			wout.writeContextObliviousEmbed(contextObliviousEmbed);
			System.out.println("+++Generated CCA Embedddings for training data+++\n");
		}
	}

	public static HashMap<String,Integer> deserializeCorpusIntMapped(Options opt) throws ClassNotFoundException{
		File f= new File(opt.serializeCorpus);
		HashMap<String,Integer> corpus_intM=null;
		
		try{
			
			ObjectInput c_intM=new ObjectInputStream(new FileInputStream(f));
			corpus_intM=(HashMap<String,Integer>)c_intM.readObject();
			
			System.out.println("=======De-serialized the CPCA Corpus Int Mapping=======");
		}
		catch (IOException ioe){
			System.out.println(ioe.getMessage());
		}
		
		return corpus_intM;
		
	} 

	public static Object[] deserializeCCAVariantsRun(Options opt) throws ClassNotFoundException{
		
		Object[] matrixObj =new Object[2];
		
		String contextDict=opt.serializeRun+"Context";
		File fContext= new File(contextDict);
		
		String eigDict=opt.serializeRun+"Eig";
		File fEig= new File(eigDict);
		
		
		Matrix eigDictMat=null,contextDictMat=null;
		
		
		try{
			
			ObjectInput cpcaEig=new ObjectInputStream(new FileInputStream(fEig));
			ObjectInput cpcaContext=new ObjectInputStream(new FileInputStream(fContext));
			
			eigDictMat=(Matrix)cpcaEig.readObject();
			contextDictMat=(Matrix)cpcaContext.readObject();	
			
			System.out.println("=======De-serialized the CCA Variant Run=======");
		}
		catch (IOException ioe){
			System.out.println(ioe.getMessage());
		}
		matrixObj[0]=(Object)eigDictMat;
		matrixObj[1]=(Object)contextDictMat;
		
		return matrixObj;
		
	}


	
	
}