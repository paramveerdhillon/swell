package edu.upenn.cis.swell.Runs;

/**
 * ver: 1.0
 * @author paramveer dhillon.
 *
 * last modified: 09/04/13
 * please send bug reports and suggestions to: dhillon@cis.upenn.edu
 */


import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import Jama.Matrix;
import edu.upenn.cis.swell.IO.Options;
import edu.upenn.cis.swell.MathUtils.SVDTemplates;
import edu.upenn.cis.swell.SpectralRepresentations.LSARepresentation;

public class LSARun implements Serializable {
	private Options _opt;
	private LSARepresentation _lsaR; 
	private int _num_hidden;
	private Matrix dictMatrix=null;
	static final long serialVersionUID = 42L;
	private int _numDocsInCorpus=0;
	
	public LSARun(Options opt, LSARepresentation lsaR,ArrayList<ArrayList<Integer>> all_Docs){
		_opt=opt;
		_lsaR=lsaR;
		_num_hidden=opt.hiddenStateSize;
		_numDocsInCorpus=all_Docs.size();
		
		dictMatrix=new Matrix(_opt.vocabSize+1,_num_hidden);
		System.out.println("Entering Compute LSA");
		computeLSA(_lsaR);
			
		//writeStats();
	}

	
	private void computeLSA(LSARepresentation _lsaR2) {
		
		 long startTime = System.currentTimeMillis();
		_lsaR2.populateTermDocMatrix();
		 long endTime = System.currentTimeMillis();
		System.out.println("===Time taken for populating term doc Matrix: "+(endTime-startTime)/1000.0/60.0+" mins===");
		
		SVDTemplates svdT=new SVDTemplates(_opt,_numDocsInCorpus);
		
		dictMatrix=svdT.computeSVD_Tropp(_lsaR2.getTermDocMatrix(),_lsaR2.getOmegaMatrix());
					
	}
	
	
	
	
	
	
	
	public Matrix getTermDict(){
		return dictMatrix;
	}
	
	
	public void serializeLSARun(){
		File f= new File(_opt.serializeRun);
		
		try{
			ObjectOutput lsaR=new ObjectOutputStream(new FileOutputStream(f));
			lsaR.writeObject(dictMatrix);
			lsaR.flush();
			lsaR.close();
			
			System.out.println("=======Serialized the LSA Run=======");
		}
		catch (IOException ioe){
			System.out.println(ioe.getMessage());
		}
		
	}
	
/*	private void writeStats() {
		BufferedWriter writer=null;
		double sSum=0;
		try {
			writer=new BufferedWriter(new OutputStreamWriter(new FileOutputStream("Output_Files/LSAStats"),"UTF8"));
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		try {
			
			
			//writer.write("Rank after SVD: "+_infoStr+"\n\n");
			writer.write("Eigenvalues in increasing order:\n");
			for (int i=0;i<s.length;i++){
				Double d=new Double(s[i]);
				writer.write(d.toString()+"\n");
				sSum+=d.doubleValue();
				
			}
			writer.write("\n\nNormalized Eigenvalues in increasing order:\n");
			for (int i=0;i<s.length;i++){
				Double d=new Double(s[i]/sSum);
				writer.write(d.toString()+"\n");
				
			}
			
			writer.close();
			
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	} */

}
