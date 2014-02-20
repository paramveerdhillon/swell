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

import Jama.Matrix;
import edu.upenn.cis.swell.MathUtils.CenterScaleNormalizeUtils;
import edu.upenn.cis.swell.SpectralRepresentations.CCARepresentation;
import edu.umbc.cs.maple.utils.JamaUtils;


public class CCAWriter extends WriteDataFile implements EmbeddingWriter {


	private BufferedWriter writer,writer1; 
	private Object[] _matrices=new Object[3];
	private ReadDataFile _rin;
	JamaUtils jutils;
	CenterScaleNormalizeUtils _utils;
	
	public CCAWriter(Options opt,ArrayList<ArrayList<Integer>> all_Docs, Object[] matrices, ReadDataFile rin) {
		super(opt, all_Docs);
		_rin=rin;
		_matrices=matrices;
		jutils=new JamaUtils();
	}
	
	public CCAWriter(Options opt,CCARepresentation ccaRep, Object[] matrices, ReadDataFile rin, CenterScaleNormalizeUtils utils) {
		super(opt,ccaRep);
		_rin=rin;
		_matrices=matrices;
		jutils=new JamaUtils();
		_utils=utils;
	}
	public void writeEigenDict() throws IOException{
		//Matrix eigenDict=_utils.center_and_scale((Matrix)_matrices[2]);
		Matrix eigenDict=(Matrix)_matrices[2];
		double[][] eigenDictArr=eigenDict.getArray();
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
			/*
			for (int j=0; j<_opt.hiddenStateSize;j++){
				
				if ( j != _opt.hiddenStateSize-1){
					writer.write(Double.toString(eigenDictArr[i][j]));
					writer.write(' ');
				}
				else{
					writer.write(Double.toString(eigenDictArr[i][j]));
					writer.write('\n');
				}
			}
			*/
			
			///
			double entry=0;
			for (int j=0; j<_opt.hiddenStateSize;j++){
				entry =eigenDictArr[i][j];
				if (_opt.logTrans){
					System.out.println("In Log Transform");
					if (eigenDictArr[i][j] >0)
						entry = Math.log(eigenDictArr[i][j]);
					else
						entry = -1*Math.log(Math.abs(eigenDictArr[i][j]));
				}
				if (_opt.sqRootTrans){
					System.out.println("In Square Root Transform");
					if (eigenDictArr[i][j] >0)
						entry = Math.sqrt(eigenDictArr[i][j]);
					else
						entry = -1*Math.log(Math.sqrt(Math.abs(eigenDictArr[i][j])));
				}
				
				
				if ( j != _opt.hiddenStateSize-1){
					
					
					writer.write(Double.toString(entry));
					writer.write(' ');
				}
				else{
					writer.write(Double.toString(entry));
					writer.write('\n');
				}
			}
			
			
			
		}
		
		writer.close();
		
	}
	
	public void writeEigenDictRandom() throws IOException{
		//Matrix eigenDict=_utils.center_and_scale((Matrix)_matrices[2]);
		Matrix eigenDict=(Matrix)_matrices[2];
		double[][] eigenDictArr=eigenDict.getArray();
		ArrayList<String> vocab=_rin.getSortedWordListString();
		
		Random r=new Random();
		try {
			writer=new BufferedWriter(new OutputStreamWriter(new FileOutputStream(_opt.eigendictName+"Random"),"UTF8"));
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
	
	public void writeLREigVectors() throws IOException{
		Matrix eigenDictL=(Matrix)_matrices[0];
		Matrix eigenDictR=(Matrix)_matrices[1];
		double[][] eigenDictArrL=eigenDictL.getArray();
		double[][] eigenDictArrR=eigenDictR.getArray();
		
		try {
			writer=new BufferedWriter(new OutputStreamWriter(new FileOutputStream(_opt.lSVecName),"UTF8"));
			writer1=new BufferedWriter(new OutputStreamWriter(new FileOutputStream(_opt.rSVecName),"UTF8"));
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
		for (int i=0;i<_opt.smoothArray.size()*_opt.hiddenStateSize;i++){
			for (int j=0; j<_opt.hiddenStateSize/2;j++){

				if ( j != (_opt.hiddenStateSize/2)-1){
					writer.write(Double.toString(eigenDictArrL[i][j]));
					writer.write(' ');
					
					writer1.write(Double.toString(eigenDictArrR[i][j]));
					writer1.write(' ');

				}
				else{
					writer.write(Double.toString(eigenDictArrL[i][j]));
					writer.write('\n');
					
					writer1.write(Double.toString(eigenDictArrR[i][j]));
					writer1.write('\n');
				}
			}
		}
		writer.close();
		writer1.close();
		
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
				//Token tok =doc.get(tok_idx++);
				writer.write(_rin.getTokForIntTrain(idx++));
				//System.out.println(_rin.getTokForIntTrain(idx-1));
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
			//writer.write('\n');
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
		
		try {
			writer=new BufferedWriter(new OutputStreamWriter(new FileOutputStream(_opt.contextOblEmbed+"Random"),"UTF8"));
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
		Random r=new Random();
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
