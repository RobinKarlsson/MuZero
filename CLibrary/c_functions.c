#include <stdio.h>
#include <math.h>
#include <stdbool.h>

double normalize(double x, float max, float min) {
	if (max > min) {
		return (x - min) / (max - min);
	}
	return x;
}

double calc_pb_c(double parent_visit_count, double subnode_visit_count, float pb_c_base, float pb_c_init) {
	double pb_c = (1 + parent_visit_count + pb_c_base) / pb_c_base;
	pb_c = log(pb_c) + pb_c_init;
	pb_c = pb_c * (sqrt(parent_visit_count) / (1 + subnode_visit_count));
	return pb_c;
}

double ucb_score(double pb_c, double subnode_prior, float subnode_discount, float subnode_reward, double subnode_value, int expanded, float max, float min) {
	double score = pb_c * subnode_prior;
	
	if(expanded) {
		double qvalue = subnode_discount * subnode_value + subnode_reward;
		
		if(qvalue < 0) {
			qvalue = 0;
		} else if(qvalue > 1) {
			qvalue = 1;
		}
		
		score = score + normalize(qvalue, max, min);
	}
	
	return score;
}
