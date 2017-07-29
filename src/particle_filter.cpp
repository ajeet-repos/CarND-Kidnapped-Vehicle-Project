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

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 100;
	default_random_engine gen;

	normal_distribution<double> noise_x(0, std[0]);
	normal_distribution<double> noise_y(0, std[1]);
	normal_distribution<double> noise_theta(0, std[2]);

	// Initializing all particles to teh first position and setting all weights to 1
	for (int i = 0; i < num_particles; i++) {

		Particle new_particle;
		new_particle.id = i;
		new_particle.x = x;
		new_particle.y = y;
		new_particle.theta = theta;
		new_particle.weight = 1.0;

		// Adding some Gaussian noise
		new_particle.x += noise_x(gen);
		new_particle.y += noise_y(gen);
		new_particle.theta += noise_theta(gen);

		// Adding to vectors
		particles.push_back(new_particle);
		weights.push_back(new_particle.weight);

	}

	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;

	normal_distribution<double> noise_x(0, std_pos[0]);
	normal_distribution<double> noise_y(0, std_pos[1]);
	normal_distribution<double> noise_theta(0, std_pos[2]);

	// safe gaurd
	if (fabs(yaw_rate) < 0.0001) {
		yaw_rate = 0.0001;
	}

	// caluculating new state with noise
	for (auto&& particle : particles) {

		particle.x += (velocity / yaw_rate) * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta)) + noise_x(gen);
        particle.y += (velocity / yaw_rate) * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t)) + noise_y(gen);
        particle.theta += yaw_rate * delta_t + noise_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (int i = 0; i < observations.size(); i++) {

		LandmarkObs obs = observations[i];

		double min_distance = INFINITY;
		int min_i = -1;	// closest landmark id

		for (int j = 0; j < predicted.size(); j++) {
			LandmarkObs pred_measur = predicted[j];

			double distance = dist(obs.x, obs.y, pred_measur.x, pred_measur.y);
			if (distance < min_distance) {
				min_distance = distance;
				min_i = pred_measur.id;
			}
		}

		observations[i].id = min_i;
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	default_random_engine gen;

	// Get coordinates of each particle
	for (int i=0; i<num_particles; i++){

		double x = particles[i].x;
		double y = particles[i].y;
		double theta = particles[i].theta;

		// Store locations of predicted landmarks that are inside the sensor range of the particle
		vector<LandmarkObs> predicted_landmarks;

		// Get coordinates for each landmark
		for (int k=0; k<map_landmarks.landmark_list.size(); k++) {

			int l_id = map_landmarks.landmark_list[k].id_i;
			double l_x = map_landmarks.landmark_list[k].x_f;
			double l_y = map_landmarks.landmark_list[k].y_f;
			LandmarkObs current_landmark = {l_id, l_x, l_y};

			// Choose landmarks within sensor range of particle
			if (fabs(dist(l_x, l_y, x, y)) <= sensor_range){

				predicted_landmarks.push_back(current_landmark);
			}
			}

			// Store transformed points
			vector<LandmarkObs> transformed_observations;

			// Transform points from vehicle's coordinate system to map's coordinate system
			for (int j=0; j<observations.size(); j++){

				double t_x = observations[j].x*cos(theta) - observations[j].y*sin(theta) + x;
				double t_y = observations[j].x*sin(theta) + observations[j].y*cos(theta) + y;

				// Store id and coordinates for transformed point
				LandmarkObs transformed_observation;
				transformed_observation.id = observations[j].id;
				transformed_observation.x = t_x;
				transformed_observation.y = t_y;


				transformed_observations.push_back(transformed_observation);

			}

			// Data Associations
			dataAssociation(predicted_landmarks, transformed_observations);

			double total_weight = 1.0;
			weights[i] = 1.0;
			vector<int> associations_vec;
			vector<double> sense_x_vec;
			vector<double> sense_y_vec;



			// Get coordinates of each transformed observation
			for (int l=0; l<transformed_observations.size(); l++){

				int o_id = transformed_observations[l].id;
				double o_x = transformed_observations[l].x;
				double o_y = transformed_observations[l].y;
				double p_x;
				double p_y;


				// Get coordinates of the predicted landmark for the transformed observation
				for (int m=0; m<predicted_landmarks.size(); m++){

					if (predicted_landmarks[m].id == o_id){

						p_x = predicted_landmarks[m].x;
						p_y = predicted_landmarks[m].y;
					}
				}

				// Calculate each weight using a mult-variate Gaussian distribution
				double w = (1/(2*M_PI*std_landmark[0]*std_landmark[1])) * exp(-(pow(p_x-o_x, 2)/(2*pow(std_landmark[0],2)) +
							pow(p_y-o_y, 2)/(2*pow(std_landmark[1],2)) - (2*(p_x-o_x)*(p_y-o_y)/(sqrt(std_landmark[0])*sqrt(std_landmark[1])))));

			total_weight *= w;
			associations_vec.push_back(o_id);
			sense_x_vec.push_back(o_x);
			sense_y_vec.push_back(o_y); 

		}

		particles[i].weight = total_weight;
		weights[i] = total_weight;

		SetAssociations(particles[i], associations_vec, sense_x_vec, sense_y_vec);
		predicted_landmarks.clear();

	}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	default_random_engine gen;

	discrete_distribution<int> d(weights.begin(), weights.end());
	vector<Particle> new_particles;

	for (int i = 0; i < num_particles; i++) {
		new_particles.push_back(particles[d(gen)]);
	}

	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

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
