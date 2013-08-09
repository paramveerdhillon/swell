package edu.upenn.cis.SpectralLearning.Runs;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.io.OutputStreamWriter;
import java.io.Serializable;
import java.io.UnsupportedEncodingException;

import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix2D;
import Jama.Matrix;
import edu.upenn.cis.SpectralLearning.IO.Options;
import edu.upenn.cis.SpectralLearning.MathUtils.SVDTemplates;
import edu.upenn.cis.SpectralLearning.SpectralRepresentations.ContextPCANGramsRepresentation;

public class ContextPCANGramsRun implements Serializable {

	private Options _opt;
	private ContextPCANGramsRepresentation _cpcaR; 
	private Matrix dictMatrix,dictMatrixContext;
	private int dim2=0;
	static final long serialVersionUID = 42L;
	Object[] _allDocs;
	double[] s;
	public ContextPCANGramsRun(Options opt, ContextPCANGramsRepresentation cpcaR,Object[] all_Docs){
		_opt=opt;
		_cpcaR=cpcaR;
		_allDocs =all_Docs;
		System.out.println("+++Entering Context PCA n-grams Compute+++");
		computeCPCA(_cpcaR);
			
		writeStats();
	}


	

	private void computeCPCA(ContextPCANGramsRepresentation _cpcaR2) {
		
		if(_opt.depbigram)
			_cpcaR2.computeLRContextMatrices();
		else
			_cpcaR2.computeLRContextMatricesSingleVocab();
		
		SparseDoubleMatrix2D C=_cpcaR2.getContextMatrix();
		SparseDoubleMatrix2D CT=_cpcaR2.getContextMatrixT();
		SVDTemplates svdTC;
		
		svdTC=new SVDTemplates(_opt,dim2);
		System.out.println("+++Generated the Context Matrix+++");
		
		dictMatrix=svdTC.computeSVD_Tropp(C, _cpcaR2.getOmegaMatrix(C.columns()),C.columns());
		s=svdTC.getSingularVals();
		dictMatrixContext=svdTC.computeSVD_Tropp(CT, _cpcaR2.getOmegaMatrix(CT.columns()),CT.columns());
			
	}
	
	
	private void writeStats() {
		BufferedWriter writer=null;
		
		double sSum=0;
		try {
			writer=new BufferedWriter(new OutputStreamWriter(new FileOutputStream("Output_Files/ContextPCANGramsStats"),"UTF8"));
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		try {
			
			
			writer.write("Eigenvalues in decreasing order:\n");
			for (int i=0;i<s.length;i++){
				Double d=new Double(s[i]);
				writer.write(d.toString()+"\n");
				sSum+=d.doubleValue();
				
			}
			writer.write("\n\nNormalized Eigenvalues in decreasing order:\n");
			for (int i=0;i<s.length;i++){
				Double d=new Double(s[i]/sSum);
				writer.write(d.toString()+"\n");
				
			}
			
			writer.close();
			
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	} 

	
	
	public void serializeContextPCANGramsRun() {
		
		String contextDict=_opt.serializeRun+"Context";
		String eigDict=_opt.serializeRun+"Eig";
		File fEig= new File(eigDict);
		File fContext= new File(contextDict);
		
		
		try{
			ObjectOutput cpcaEig=new ObjectOutputStream(new FileOutputStream(fEig));
			
			cpcaEig.writeObject(dictMatrix);
			cpcaEig.flush();
			cpcaEig.close();
			
			ObjectOutput cpcaContext=new ObjectOutputStream(new FileOutputStream(fContext));
			
			cpcaContext.writeObject(dictMatrixContext);
			cpcaContext.flush();
			cpcaContext.close();
			
				System.out.println("=======Serialized the Context PCA NGrams Run=======");
		}
		catch (IOException ioe){
			System.out.println(ioe.getMessage());
		}
		
		
	}
}
