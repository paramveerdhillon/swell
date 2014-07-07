package edu.upenn.cis.swell.MathUtils;

/**
 * ver: 1.0
 * @author paramveer dhillon.
 *
 * last modified: 09/04/13
 * please send bug reports and suggestions to: dhillon@cis.upenn.edu
 */


import java.io.Serializable;

import Jama.Matrix;
import edu.upenn.cis.swell.IO.Options;

public class CenterScaleNormalizeUtils implements Serializable {

	Options _opt;	
	static final long serialVersionUID = 42L;
	
	public CenterScaleNormalizeUtils(Options opt){
		_opt=opt;
	}
	
public Matrix center_and_scale(Matrix eigen_dict){
		
		double[][] eigenFeatDict=new double[_opt.vocabSize+1][_opt.hiddenStateSize];
		Object[] mean_sd;
		double[] _means=new double[eigen_dict.getColumnDimension()];
		double[] _sds=new double[eigen_dict.getColumnDimension()];
		
		
		mean_sd=compute_mean_std(eigen_dict);
		_means=(double[])mean_sd[0];
		_sds=(double[])mean_sd[1];
		
		eigenFeatDict=eigen_dict.getArray();
		
		for (int i=0; i<eigen_dict.getColumnDimension();i++){
			for(int j=0;j<eigen_dict.getRowDimension();j++){
			eigenFeatDict[j][i]-=_means[i];
			eigenFeatDict[j][i]/=_sds[i];
		}
		}
			Matrix new_eigen_dict=new Matrix(eigenFeatDict);
			
			
			
			return new_eigen_dict;
		
		}


public Matrix sqrtTransform(Matrix eigen_dict){
	
	double[][] eigenFeatDict=new double[_opt.vocabSize+1][_opt.hiddenStateSize];
	
	eigenFeatDict=eigen_dict.getArray();
	
	for (int i=0; i<eigen_dict.getColumnDimension();i++){
		for(int j=0;j<eigen_dict.getRowDimension();j++){
			if(eigenFeatDict[j][i] >=0){
				eigenFeatDict[j][i]= Math.sqrt(eigenFeatDict[j][i]);
			}
			else{
				eigenFeatDict[j][i]= -1*Math.sqrt(-1*eigenFeatDict[j][i]);
			}
	}
	}
		Matrix new_eigen_dict=new Matrix(eigenFeatDict);
		
		
		
		return new_eigen_dict;
	
	}

public Matrix normalize(Matrix eigen_dict){
	
	double[][] eigenFeatDict=new double[eigen_dict.getRowDimension()][eigen_dict.getColumnDimension()];
	
	eigenFeatDict=eigen_dict.getArray();
	
	double[] maxInThisDimension= new double[eigen_dict.getRowDimension()];
	
	double arr=0;
	
	for (int i=0; i<eigen_dict.getRowDimension();i++){
		arr=0;
		for(int j=0;j<eigen_dict.getColumnDimension();j++){
			arr=eigenFeatDict[i][j];
			if(maxInThisDimension[i]<Math.abs(arr))
				maxInThisDimension[i]=Math.abs(arr);
		}
	}
	
	
	for (int i=0; i<eigen_dict.getRowDimension();i++){
		for(int j=0;j<eigen_dict.getColumnDimension();j++){
				eigenFeatDict[i][j]= eigenFeatDict[i][j]/maxInThisDimension[i];
			}		
	}
	
		Matrix new_eigen_dict=new Matrix(eigenFeatDict);	
		return new_eigen_dict;
	
	}




public Matrix center(Matrix eigen_dict){
	
	double[][] eigenFeatDict=new double[_opt.vocabSize+1][_opt.hiddenStateSize];
	Object[] mean_sd;
	double[] _means=new double[eigen_dict.getColumnDimension()];
	
	
	mean_sd=compute_mean_std(eigen_dict);
	_means=(double[])mean_sd[0];
	
	eigenFeatDict=eigen_dict.getArray();
	
	for (int i=0; i<eigen_dict.getColumnDimension();i++){
		for(int j=0;j<eigen_dict.getRowDimension();j++){
		eigenFeatDict[j][i]-=_means[i];
	}
	}
		Matrix new_eigen_dict=new Matrix(eigenFeatDict);
		
		
		
		return new_eigen_dict;
	
	}

   public double cantorPairingMap (int i, int j){ //See the wiki article
	
	   double num=(0.5*(i+j)*(i+j+1)) + j;
	   return num;
   }
   
   public double[] cantorPairingInverseMap (double k){ //See the wiki article
	
	   double w= Math.floor((Math.sqrt((8*k)+1)-1)/2);
	   double t= ((w*w)+ w)/2;
	   double[] vals=new double[2];
	   
	   vals[1]=  k-t;
	   vals[0]=w-vals[1];
	   return vals;
   }
   
   
	public Object[] compute_mean_std(Matrix eigen_dict) {
		Object[] mean_std=new Object[2];
		
		double sumCounter=0,sumSqCounter=0;
		double[] means=new double[eigen_dict.getColumnDimension()];
		double[] MSEs=new double[eigen_dict.getColumnDimension()];
		double[] sds=new double[eigen_dict.getColumnDimension()];
		
		for (int i=0;i<eigen_dict.getColumnDimension();i++){
			for(int j=0;j<eigen_dict.getRowDimension();j++){
				sumCounter+=eigen_dict.get(j, i);
				sumSqCounter+=eigen_dict.get(j, i)*eigen_dict.get(j, i);
			}
			means[i]=sumCounter/eigen_dict.getRowDimension();
			MSEs[i]=sumSqCounter/eigen_dict.getRowDimension();
			sumCounter=0;
			sumSqCounter=0;
		}
		for (int k=0; k<means.length;k++){
			sds[k]=Math.sqrt(MSEs[k]- (means[k]*means[k]));
		}
		
		mean_std[0]=means;
		mean_std[1]=sds;
		return mean_std;
	}

	
}
