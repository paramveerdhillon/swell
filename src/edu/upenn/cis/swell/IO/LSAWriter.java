package edu.upenn.cis.swell.IO;

import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Random;

import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import Jama.Matrix;
import edu.upenn.cis.swell.Data.Corpus;
import edu.upenn.cis.swell.Data.Document;
import edu.upenn.cis.swell.MathUtils.MatrixFormatConversion;

public class LSAWriter extends WriteDataFile implements EmbeddingWriter {

	private BufferedWriter writer; 
	private ReadDataFile _rin;
	private Matrix _eigenD=null;
	ArrayList<ArrayList<Integer>> _allDocs;
	
	public LSAWriter(Options opt, ArrayList<ArrayList<Integer>> all_Docs, Matrix eigenD, ReadDataFile rin) {
		super(opt, all_Docs);
		_allDocs =all_Docs;
		_eigenD=eigenD;
		_rin=rin;
		
	}
	
	public void writeEigenDict() throws IOException{
		DenseDoubleMatrix2D dictRMatrix=MatrixFormatConversion.createDenseMatrixCOLT(_eigenD);

		ArrayList<String> vocab=_rin.getSortedWordListString();
		try {
			writer=new BufferedWriter(new OutputStreamWriter(new FileOutputStream(_opt.eigendictName),"UTF8"));
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
		for (int i=0; i<=vocab.size(); i++) {
			
			if (i==0){
				writer.write("<OOV>");
				writer.write(' ');
			}
			else{
				writer.write(vocab.get(i-1));
				writer.write(' ');
			}
			for (int j=0; j<_opt.hiddenStateSize;j++){
				
				if ( j != _opt.hiddenStateSize-1){
					writer.write(Double.toString(dictRMatrix.get(i, j)));
					writer.write(' ');
				}
				else{
					writer.write(Double.toString(dictRMatrix.get(i, j)));
					writer.write('\n');
				}
			}
			
		}
		
		writer.close();
		
	}
	
	
	public void writeEigenDictRandom() throws IOException{
		
		Random r= new Random();
		ArrayList<String> vocab=_rin.getSortedWordListString();
		try {
			writer=new BufferedWriter(new OutputStreamWriter(new FileOutputStream(_opt.eigendictName),"UTF8"));
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
		for (int i=0; i<=vocab.size(); i++) {
			
			if (i==0){
				writer.write("<OOV>");
				writer.write(' ');
			}
			else{
				writer.write(vocab.get(i-1));
				writer.write(' ');
			}
			for (int j=0; j<_opt.hiddenStateSize;j++){
				
				if ( j != _opt.hiddenStateSize-1){
					writer.write(Double.toString(r.nextGaussian()));
					writer.write(' ');
				}
				else{
					writer.write(Double.toString(r.nextGaussian()));
					writer.write('\n');
				}
			}
			
		}
		
		writer.close();
		
	}
	
	
	public void writeContextObliviousEmbed(Matrix contextObliviousEmbed) throws IOException {
		int i=0,idxDoc=0,idx=0;
		
		try {
			writer=new BufferedWriter(new OutputStreamWriter(new FileOutputStream(_opt.contextOblEmbed),"UTF8"));
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
		//Matrix m1=jutils.colsum(contextObliviousEmbed);
	
	while(idxDoc<_allDocs.size()){	
			int tok_idx=0;	
			ArrayList<Integer> doc=_allDocs.get(idxDoc++);
			
			while(tok_idx<doc.size()){
				writer.write(_rin.getTokForIntTrain(idx++));
				writer.write(' ');
				for (int j=0;j<_opt.hiddenStateSize;j++){
					
					if ( j != (_opt.hiddenStateSize)-1){
						writer.write(Double.toString(contextObliviousEmbed.get(i, j)));
						writer.write(' ');
					}
					else{
						writer.write(Double.toString(contextObliviousEmbed.get(i, j)));
						writer.write('\n');
					}
				}
				i++;
				tok_idx++;
			}
	}
		writer.close();
		
		
	}

	public void writeContextObliviousEmbedRandom() throws IOException {
		int i=0,idxDoc=0,idx=0;
		Random r= new Random();
		try {
			writer=new BufferedWriter(new OutputStreamWriter(new FileOutputStream(_opt.contextOblEmbed),"UTF8"));
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
		//Matrix m1=jutils.colsum(contextObliviousEmbed);
	
	while(idxDoc<_allDocs.size()){	
			int tok_idx=0;	
			ArrayList<Integer> doc=_allDocs.get(idxDoc++);
			
			while(tok_idx<doc.size()){
				writer.write(_rin.getTokForIntTrain(idx++));
				writer.write(' ');
				for (int j=0;j<_opt.hiddenStateSize;j++){
					
					if ( j != (_opt.hiddenStateSize)-1){
						writer.write(Double.toString(r.nextGaussian()));
						writer.write(' ');
					}
					else{
						writer.write(Double.toString(r.nextGaussian()));
						writer.write('\n');
					}
				}
				i++;
				tok_idx++;
			}
	}
		writer.close();
		
		
	}

	
	
}
