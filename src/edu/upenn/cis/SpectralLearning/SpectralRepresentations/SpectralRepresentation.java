package edu.upenn.cis.SpectralLearning.SpectralRepresentations;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

import Jama.Matrix;
import edu.upenn.cis.SpectralLearning.Data.Corpus;
import edu.upenn.cis.SpectralLearning.Data.Document;
import edu.upenn.cis.SpectralLearning.IO.Options;
import edu.upenn.cis.SpectralLearning.MathUtils.CenterScaleNormalizeUtils;

public abstract class SpectralRepresentation implements Serializable{
	
	protected int _num_hidden=50;
	private int _vocab_size=30000;
	protected long _num_tokens=-1;
	static final long serialVersionUID = 42L;
	protected Options _opt;
	protected Matrix eigenFeatDictMatrix;
	CenterScaleNormalizeUtils mathUtils;
	private Corpus _corpus;
	
	public SpectralRepresentation(Options opt, long numTok){
		_opt=opt;
		_num_hidden=_opt.hiddenStateSize;
		_vocab_size=_opt.vocabSize;
		_num_tokens=numTok;
		
		mathUtils=new CenterScaleNormalizeUtils(_opt);
		
		initialize();
	}
	
	protected void initialize(){
		
		Random r= new Random();
		double[][] eigenFeatDict= new double[_vocab_size+1][_num_hidden];
		for (int i=0;i<_vocab_size+1;i++){
			for (int j=0;j<_num_hidden;j++)
				eigenFeatDict[i][j]=r.nextGaussian();
		}
		
		
		eigenFeatDictMatrix= new Matrix(eigenFeatDict); //v times k matrix
		eigenFeatDictMatrix=mathUtils.center_and_scale(eigenFeatDictMatrix);
	}
	
	
	
	
	
	
	public Matrix getEigenFeatDict(){
		return eigenFeatDictMatrix;
	}
	
	public void setEigenFeatDict(Matrix eigenFeatDictMatrix){
		this.eigenFeatDictMatrix=eigenFeatDictMatrix;
	}
}


