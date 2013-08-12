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
import edu.upenn.cis.SpectralLearning.Data.Corpus;
import edu.upenn.cis.SpectralLearning.IO.CCAWriter;
import edu.upenn.cis.SpectralLearning.IO.Options;
import edu.upenn.cis.SpectralLearning.IO.ReadDataFile;
import edu.upenn.cis.SpectralLearning.MathUtils.CenterScaleNormalizeUtils;
import edu.upenn.cis.SpectralLearning.Runs.CCARun;
import edu.upenn.cis.SpectralLearning.SpectralRepresentations.CCARepresentation;

public class LRMVLEmbed implements Serializable {
	
	static final long serialVersionUID = 42L;
	
public static void main(String[] args) throws Exception{
		
	
		ArrayList<ArrayList<Integer>> all_Docs;
		ArrayList<Integer> docSize;
		long numTokens=0;
		ReadDataFile rin;
		Corpus corpus;
		CCARepresentation ccaRep;
		CCARun ccaRun;
		CCAWriter wout;
		Matrix contextSpecificEmbed,contextObliviousEmbed;
		HashMap<String,Integer> corpusInt=new HashMap<String,Integer>();
		HashMap<String,Integer> corpusIntOldMapping=new HashMap<String,Integer>();
		Object[] matrices=new Object[3];
		CenterScaleNormalizeUtils utils=null;
	
		Options opt=new Options(args);
		
		if(opt.algorithm==null){
			System.out.println("WARNING: YOU NEED TO SPECIFY A VALID ALGORITHM NAME AS algorithm:");
		}
		utils=new CenterScaleNormalizeUtils(opt);
		if (opt.trainUnlab)
		{
			System.out.println("+++Inducing LR-MVL Embedddings from unlabeled data+++\n");
			
			all_Docs=new ArrayList<ArrayList<Integer>>();
			docSize=new ArrayList<Integer>();
			rin=new ReadDataFile(opt);
			corpusInt= rin.convertAllDocsInt(0);
			all_Docs=rin.readAllDocs(0);
			docSize=rin.getDocSizes();
			numTokens=rin.getNumTokens();
			rin.serializeCorpusIntMapped();
			System.out.println("+++Read Docs.+++\n");
			//corpus=new Corpus(all_Docs,docSize,opt);
			System.out.println("+++Read a total of "+numTokens+" tokens from "+docSize.size()+" documents+++");
			System.out.println("+++Created and Serialized Corpus+++\n");
			ccaRep= new CCARepresentation(opt, numTokens,rin,all_Docs);
			ccaRun=new CCARun(opt,ccaRep);
			ccaRun.serializeCCARun();
			matrices=deserializeCCARun(opt);
			wout=new CCAWriter(opt,all_Docs,matrices,rin,utils);
			wout.writeEigenDict();
			wout.writeLREigVectors();
			
			if (opt.randomBaseline){
				wout.writeEigenDictRandom();
			}
			
			
			System.out.println("+++LR-MVL Embedddings Induced+++\n");	
		}
		if (opt.train){
			System.out.println("+++Generating LR-MVL Embedddings for training data+++\n");
			all_Docs=new ArrayList<ArrayList<Integer>>();
			docSize=new ArrayList<Integer>();
			corpusIntOldMapping=deserializeCorpusIntMapped(opt);
			rin=new ReadDataFile(opt);
			rin.setCorpusIntMapped(corpusIntOldMapping);
			all_Docs=rin.readAllDocs(1);
			docSize=rin.getDocSizes();
			numTokens=rin.getNumTokens();
			//corpus=new Corpus(all_Docs,docSize,opt);
			matrices=deserializeCCARun(opt);
			ccaRep= new CCARepresentation(opt, numTokens,rin,all_Docs);
			
			
			contextSpecificEmbed=ccaRep.generateProjectionsBySmoothing((Matrix)matrices[2], 
					(Matrix)matrices[0], (Matrix)matrices[1]);
			contextObliviousEmbed=ccaRep.getContextOblEmbeddings((Matrix)matrices[2]);
	
			wout=new CCAWriter(opt,all_Docs,matrices,rin,utils);
			wout.writeContextSpecificEmbed(contextSpecificEmbed);
			wout.writeContextObliviousEmbed(contextObliviousEmbed);
			if (opt.randomBaseline){
				wout.writeContextObliviousEmbedRandom();
			}
			
			System.out.println("+++Generated LR-MVL Embedddings for training data+++\n");
			
		}
		
		
	}


public static HashMap<String,Integer> deserializeCorpusIntMapped(Options opt) throws ClassNotFoundException{
	File f= new File(opt.serializeCorpus);
	HashMap<String,Integer> corpus_intM=null;
	
	try{
		
		ObjectInput c_intM=new ObjectInputStream(new FileInputStream(f));
		corpus_intM=(HashMap<String,Integer>)c_intM.readObject();
		
		System.out.println("=======De-serialized the CCA Run=======");
	}
	catch (IOException ioe){
		System.out.println(ioe.getMessage());
	}
	
	return corpus_intM;
	
} 

public static Object[] deserializeCCARun(Options opt) throws ClassNotFoundException{
	
	Object[] matrixObj=new Object[3];
	String leftEig=opt.serializeRun+"L";
	File fL= new File(leftEig);
	
	String rightEig=opt.serializeRun+"R";
	File fR= new File(rightEig);
	
	String eigDict=opt.serializeRun+"Eig";
	File fEig= new File(eigDict);
	
	Matrix eigDictL=null,eigDictR=null,eig_Dict=null;
	
	
	try{
		
		ObjectInput ccal=new ObjectInputStream(new FileInputStream(fL));
		ObjectInput ccar=new ObjectInputStream(new FileInputStream(fR));
		ObjectInput ccaw=new ObjectInputStream(new FileInputStream(fEig));
		
		eigDictL=(Matrix)ccal.readObject();
		eigDictR=(Matrix)ccar.readObject();
		eig_Dict=(Matrix)ccaw.readObject();
		
		System.out.println("=======De-serialized the CCA Run=======");
	}
	catch (IOException ioe){
		System.out.println(ioe.getMessage());
	}
	matrixObj[0]=(Object)eigDictL;
	matrixObj[1]=(Object)eigDictR;
	matrixObj[2]=(Object)eig_Dict;
	
	return matrixObj;
	
}

	


}
