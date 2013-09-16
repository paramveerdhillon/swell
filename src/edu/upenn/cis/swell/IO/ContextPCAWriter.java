package edu.upenn.cis.swell.IO;

/**
 * ver: 1.0
 * @author paramveer dhillon.
 *
 * last modified: 09/04/13
 * please send bug reports and suggestions to: dhillon@cis.upenn.edu
 */

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

public class ContextPCAWriter extends WriteDataFile implements EmbeddingWriter {

	private Object[] _matrices=new Object[3];
	private BufferedWriter writer;
	private ReadDataFile _rin;
	ArrayList<ArrayList<Integer>> _allDocs;
	
	public ContextPCAWriter(Options opt, ArrayList<ArrayList<Integer>> all_Docs,Object[] matrices,ReadDataFile rin) {
		super(opt, all_Docs);
		_allDocs =all_Docs;
		_matrices=matrices;
		_rin=rin;
		
	}
	
	private DenseDoubleMatrix2D createDenseMatrixCOLT(Matrix xJama) {
		DenseDoubleMatrix2D x_omega=new DenseDoubleMatrix2D(xJama.getRowDimension(),xJama.getColumnDimension());
		for (int i=0;i<xJama.getRowDimension();i++){
			for (int j=0;j<xJama.getColumnDimension();j++){
				x_omega.set(i, j, xJama.get(i, j));
			}
		}
		return x_omega;
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
	
	
	public void writeContextObliviousEmbedContext(Matrix contextObliviousEmbed) throws IOException {
		int i=0,idxDoc=0,idx=0;
		
		try {
			writer=new BufferedWriter(new OutputStreamWriter(new FileOutputStream(_opt.contextOblEmbedContext),"UTF8"));
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
	
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
		Random r =new Random();
		try {
			writer=new BufferedWriter(new OutputStreamWriter(new FileOutputStream(_opt.contextOblEmbed+"Random"),"UTF8"));
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
	
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
	
	public void writeEigContextVectors() throws IOException{
		Matrix eigenDictContext=(Matrix)_matrices[1];
		double[][] eigenDictArrContext=eigenDictContext.getArray();
		ArrayList<String> vocab=_rin.getSortedWordListString();
		String contextFile =_opt.outputDir+"contextDict"+_opt.algorithm+_opt.typeofDecomp;
		int counter=0;
		
		try {
			writer=new BufferedWriter(new OutputStreamWriter(new FileOutputStream(contextFile),"UTF8"));
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		if(_opt.typeofDecomp.equals("WvsL")||_opt.typeofDecomp.equals("WvsR")||_opt.typeofDecomp.equals("2viewWvsL")||_opt.typeofDecomp.equals("2viewWvsR"))
			counter=_opt.contextSizeOneSide;
		else
			counter=2*_opt.contextSizeOneSide;
		int c=0;
		for (int i=0; i<counter*(vocab.size()+1); i++) {
			if(i%(vocab.size()+1)==0 && i!=0)
				c++;
				
			if (i==0 || i==c*(vocab.size()+1)){
				writer.write("<OOV>");
				writer.write(' ');
			}
			else{
				if(i<=vocab.size()){
					writer.write(vocab.get(i-1));
					writer.write(' ');
				}else{
					writer.write(vocab.get(i-(c*vocab.size())-1-c));
					writer.write(' ');
				}
					
			}
			for (int j=0; j<_opt.hiddenStateSize;j++){
				if(i<=vocab.size()){
					if ( j != _opt.hiddenStateSize-1){
						writer.write(Double.toString(eigenDictArrContext[i][j]));
						writer.write(' ');
					}
					else{
						writer.write(Double.toString(eigenDictArrContext[i][j]));
						writer.write('\n');
					}
				}
				else{
					if ( j != _opt.hiddenStateSize-1){
						writer.write(Double.toString(eigenDictArrContext[i-c*vocab.size()][j]));
						writer.write(' ');
					}
					else{
						writer.write(Double.toString(eigenDictArrContext[i-c*vocab.size()][j]));
						writer.write('\n');
					}
				}
			}
		}
		writer.close();
	}

	
	public void writeEigContextVectorsRandom() throws IOException{
		Matrix eigenDictContext=(Matrix)_matrices[1];
		double[][] eigenDictArrContext=eigenDictContext.getArray();
		ArrayList<String> vocab=_rin.getSortedWordListString();
		String contextFile =_opt.outputDir+"contextDictRandom";
		
		Random r= new Random();
		int counter=0;
		
		try {
			writer=new BufferedWriter(new OutputStreamWriter(new FileOutputStream(contextFile),"UTF8"));
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		if(_opt.typeofDecomp.equals("WvsL")||_opt.typeofDecomp.equals("WvsR")||_opt.typeofDecomp.equals("2viewWvsL")||_opt.typeofDecomp.equals("2viewWvsR"))
			counter=_opt.contextSizeOneSide;
		else
			counter=2*_opt.contextSizeOneSide;
		int c=0;
		for (int i=0; i<counter*(vocab.size()+1); i++) {
			if(i%(vocab.size()+1)==0 && i!=0)
				c++;
				
			if (i==0 || i==c*(vocab.size()+1)){
				writer.write("<OOV>");
				writer.write(' ');
			}
			else{
				if(i<=vocab.size()){
					writer.write(vocab.get(i-1));
					writer.write(' ');
				}else{
					writer.write(vocab.get(i-(c*vocab.size())-1-c));
					writer.write(' ');
				}
					
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
	
	
	
	public void writeContextSpecificEmbed(Matrix contextSpecificEmbed) throws IOException {
		int i=0,idxDoc=0,idx=0;
		
		try {
			writer=new BufferedWriter(new OutputStreamWriter(new FileOutputStream(_opt.contextSpecificEmbed),"UTF8"));
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	
	while(idxDoc<_allDocs.size()){	
			int tok_idx=0;	
			ArrayList<Integer> doc=_allDocs.get(idxDoc++);
			
			while(tok_idx<doc.size()){
				writer.write(_rin.getTokForIntTrain(idx++));
				writer.write(' ');
				for (int j=0;j<_opt.hiddenStateSize*3;j++){
					
					if ( j != (3*_opt.hiddenStateSize)-1){
						writer.write(Double.toString(contextSpecificEmbed.get(i, j)));
						writer.write(' ');
					}
					else{
						writer.write(Double.toString(contextSpecificEmbed.get(i, j)));
						writer.write('\n');
					}
				}
				i++;
				tok_idx++;
			}
	}
		writer.close();
		
	}

	

	
	public void writeEigenDict() throws IOException{
		DenseDoubleMatrix2D dictLMatrix=createDenseMatrixCOLT((Matrix)_matrices[0]);
		double[][] dictL=dictLMatrix.toArray();
		ArrayList<String> vocab=_rin.getSortedWordListString();
		String eigenDict=_opt.outputDir+"eigenDict"+_opt.algorithm+_opt.typeofDecomp;
		try {
			writer=new BufferedWriter(new OutputStreamWriter(new FileOutputStream(eigenDict),"UTF8"));
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
					writer.write(Double.toString(dictL[i][j]));
					writer.write(' ');
				}
				else{
					writer.write(Double.toString(dictL[i][j]));
					writer.write('\n');
				}
			}
			
		}
		
		writer.close();
		
	}

	public void writeEigenDictRandom() throws IOException{
		Random r= new Random();
		DenseDoubleMatrix2D dictLMatrix=createDenseMatrixCOLT((Matrix)_matrices[0]);
		double[][] dictL=dictLMatrix.toArray();
		ArrayList<String> vocab=_rin.getSortedWordListString();
		String eigenDict=_opt.outputDir+"eigenDictRandom";
		try {
			writer=new BufferedWriter(new OutputStreamWriter(new FileOutputStream(eigenDict),"UTF8"));
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
	
	
}
