/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <cfloat>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[], int _num_particles) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	num_particles = _num_particles;

	normal_distribution<double> distribution_x(x, std[0]);
	normal_distribution<double> distribution_y(y, std[1]);
	normal_distribution<double> distribution_theta(theta, std[2]);

	default_random_engine rand_eng;

	for(int iParticle = 0; iParticle < num_particles; iParticle++)
	{
		Particle p;

		p.id 	= iParticle;
		p.x 	= distribution_x(rand_eng);
		p.y 	= distribution_y(rand_eng);
		p.theta = distribution_theta(rand_eng);
		p.weight = 1.0;

		particles.push_back(p);
	}

	is_initialized = true;

}

void normalize_angle(double& angle)
{
	while(angle > 2 * M_PI)
		angle -= 2 * M_PI;

	while(angle < 0.0)
		angle += 2 * M_PI;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.

	default_random_engine rand_eng;
	normal_distribution<double> noise_x(0.0, std_pos[0]);
	normal_distribution<double> noise_y(0.0, std_pos[1]);
	normal_distribution<double> noise_theta(0.0, std_pos[2]);
	const double thresh = 0.0001;

	//Nearly straight line motion causes division by zero in yaw calculation.
	//Test once outside our loop for better performance.
	if(abs(yaw_rate) > thresh)
	{
		const double vel_over_yaw = velocity / yaw_rate;
		const double yaw_rate_dt = yaw_rate * delta_t;
	
		for(auto& p : particles)
		{
			p.x += vel_over_yaw * (sin(p.theta + yaw_rate_dt) - sin(p.theta)) + noise_x(rand_eng);
			p.y += vel_over_yaw * (cos(p.theta) - cos(p.theta + yaw_rate_dt)) + noise_y(rand_eng);
			p.theta += yaw_rate_dt + noise_theta(rand_eng);

			normalize_angle(p.theta);

		}
	}
	else
	{
		//No appreciable rotation, straight line motion model
		for(auto& p : particles)
		{
			p.x += velocity * cos(p.theta) + noise_x(rand_eng);
			p.y += velocity * sin(p.theta) + noise_y(rand_eng);
			p.theta += noise_theta(rand_eng);

			normalize_angle(p.theta);
		}
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	
	for(auto& pred : predicted)
	{
		pred.dx = 0.0;
		pred.dy = 0.0;

		double closest_dist = DBL_MAX;
		LandmarkObs* pClosest = nullptr;

		for(auto& obs : observations)
		{
			//no need for sqrt. we just need the closest.
			double dist_ob_pred = dist_sqrd(obs.x, obs.y, pred.x, pred.y);

			if(dist_ob_pred < closest_dist)
			{
				closest_dist = dist_ob_pred;
				pClosest = &obs;
			}
		}

		if (pClosest != nullptr)
		{
			pClosest->id = pred.id;

			//to save time with our multi-variate normal probablity calc later,
			//we will save these terms that depend on distance from associated landmark
			pClosest->dx = pred.x - pClosest->x;
			pClosest->dy = pred.y - pClosest->y;
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution.
	//   reference: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. 

	//Some constant terms for all particles in the Multivariate-Gaussian Probability equation
	const double sigma_x = std_landmark[0];
	const double sigma_y = std_landmark[1];
	const double gauss_norm = (1.0 / (2.0 * M_PI * sigma_x * sigma_y));
	const double two_sigma_x_sq = 2.0 * sigma_x * sigma_x;
	const double two_sigma_y_sq = 2.0 * sigma_y * sigma_y;

	//clear prev weight values
	weights.clear();

	for(auto& p : particles)
	{
		//transform observations into particle coordinate system
		vector<LandmarkObs> obs_relative_to_particle;

		//improve heap fragmentation by reserving exact size needed
		obs_relative_to_particle.reserve(observations.size());

		//precalucate expensive terms
		double cos_theta = cos(p.theta);
		double sin_theta = sin(p.theta);

		LandmarkObs tm_obs;

		for(auto& obs : observations)
		{
			tm_obs.id = obs.id;
			tm_obs.x = (cos_theta * obs.x) - (sin_theta * obs.y) + p.x;
			tm_obs.y = (sin_theta * obs.x) + (cos_theta * obs.y) + p.y;

			obs_relative_to_particle.push_back(tm_obs);
		}

		//consider only landmarks in sensor range
		vector<LandmarkObs> lm_in_range;

		//improve heap fragmentation by reserving largest estimate
		lm_in_range.reserve(map_landmarks.landmark_list.size());

		//no need for abolute range, which requires a sqrt on every term.
		//we can instead work in dist squared space
		double sensor_range_squared = sensor_range * sensor_range;

		for(auto& lm : map_landmarks.landmark_list)
		{
			double dist_lm_part = dist_sqrd(lm.x_f, lm.y_f, p.x, p.y);

			if(dist_lm_part < sensor_range_squared)
			{
				LandmarkObs map_obs { lm.id_i, lm.x_f, lm.y_f};
				lm_in_range.push_back(map_obs);
			}
		}

		//associate an observation with the closest landmark
		dataAssociation(lm_in_range, obs_relative_to_particle);

		//determine the weight of the particle as the product of
		//the Multivariate-Gaussian Problabilities of dist to each observed landmark
		p.weight = 1.0;

		double exponent, weight;
		const double min_weight = 0.0001;

		p.associations.clear();
		double prob = 0.0;

		for(auto& obs : obs_relative_to_particle)
		{
			//exponent = (obs.dx * obs.dx) / two_sigma_x_sq + (obs.dy * obs.dy) / two_sigma_y_sq;
			//weight = gauss_norm * exp(-1.0 * exponent);
			//p.weight *= (weight > min_weight) ? weight : min_weight;

			//do calculation in log space to avoid numerical underflow
			//https://stats.stackexchange.com/questions/95322/problem-with-estimating-probability-using-the-multivariate-gaussian
			double scale = log( 2.0 * sigma_x * sigma_y * M_PI);
			double d0 = (obs.dx) * (obs.dx) / (sigma_x * sigma_x);
			double d1 = (obs.dy) * (obs.dy) / (sigma_y * sigma_y);
			double mvnormal_log_density = -0.5 * (scale + d0 + d1);

			prob += mvnormal_log_density;
		}

		//expoentiate once at the end
		p.weight = exp(prob);


		weights.push_back(p.weight);
	}
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight. 
	// NOTE: Uses std::discrete_distribution. reference:
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	std::random_device rd;
    std::mt19937 gen(rd());
	
	/* Not sure why I was getting an error before...
	//create an integral type from our double weight array.
	vector<long> int_weights;
	for(auto& w  : weights)
	{
		int_weights.push_back(long(w * 100000.0));
	}

	if(weights.size() != particles.size())
		cout << weights.size() << particles.size() << endl;		

    std::discrete_distribution<> d(int_weights.begin(), int_weights.end());
	*/
    std::discrete_distribution<> d(weights.begin(), weights.end());

	std::vector<Particle> prev_particles = particles;

    for(int iNewPart=0; iNewPart < num_particles; ++iNewPart) 
	{
		//sample a previous particle based on the discrete distribution
        int iPrevPart = d(gen);

		if(iPrevPart > -1 && iPrevPart < prev_particles.size())
			particles[iNewPart] = prev_particles[iPrevPart];
		else
			cout << "err: iPrevPart out of range" << iPrevPart << endl;

    }
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
