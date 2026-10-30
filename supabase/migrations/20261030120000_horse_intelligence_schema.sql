-- Schéma Horse Intelligence (généré depuis src/horse_intelligence/data/storage/database.py)
CREATE TABLE IF NOT EXISTS collection_runs (
	run_id SERIAL NOT NULL, 
	target_date DATE NOT NULL, 
	started_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	finished_at TIMESTAMP WITH TIME ZONE, 
	status VARCHAR(20) NOT NULL, 
	races_seen INTEGER, 
	races_ok INTEGER, 
	errors_json JSON, 
	PRIMARY KEY (run_id)
);
CREATE TABLE IF NOT EXISTS data_quality_reports (
	report_id SERIAL NOT NULL, 
	scope VARCHAR(40) NOT NULL, 
	scope_ref VARCHAR(100) NOT NULL, 
	run_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	status VARCHAR(30) NOT NULL, 
	checks_json JSON NOT NULL, 
	PRIMARY KEY (report_id)
);
CREATE TABLE IF NOT EXISTS data_sources (
	source_key VARCHAR(40) NOT NULL, 
	name VARCHAR(200) NOT NULL, 
	base_url VARCHAR(500), 
	kind VARCHAR(40), 
	enabled BOOLEAN NOT NULL, 
	status_audit VARCHAR(40), 
	last_status VARCHAR(40), 
	last_checked_at TIMESTAMP WITH TIME ZONE, 
	last_message TEXT, 
	PRIMARY KEY (source_key)
);
CREATE TABLE IF NOT EXISTS detected_patterns (
	pattern_id SERIAL NOT NULL, 
	definition_json JSON NOT NULL, 
	discovery_start DATE, 
	discovery_end DATE, 
	n_obs INTEGER, 
	effect_size FLOAT, 
	ci_low FLOAT, 
	ci_high FLOAT, 
	p_value FLOAT, 
	q_value_fdr FLOAT, 
	oos_effect FLOAT, 
	oos_ci_low FLOAT, 
	oos_ci_high FLOAT, 
	status VARCHAR(30) NOT NULL, 
	history_json JSON, 
	PRIMARY KEY (pattern_id)
);
CREATE TABLE IF NOT EXISTS feature_sets (
	feature_set_id SERIAL NOT NULL, 
	version VARCHAR(40) NOT NULL, 
	code_hash VARCHAR(64) NOT NULL, 
	definition_json JSON NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (feature_set_id), 
	UNIQUE (version)
);
CREATE TABLE IF NOT EXISTS horses (
	horse_id SERIAL NOT NULL, 
	canonical_name VARCHAR(200) NOT NULL, 
	display_name VARCHAR(200) NOT NULL, 
	birth_year INTEGER, 
	breed VARCHAR(80), 
	sex_last VARCHAR(20), 
	PRIMARY KEY (horse_id), 
	CONSTRAINT uq_horse_identity UNIQUE (canonical_name, birth_year)
);
CREATE TABLE IF NOT EXISTS jockeys (
	jockey_id SERIAL NOT NULL, 
	canonical_name VARCHAR(200) NOT NULL, 
	display_name VARCHAR(200) NOT NULL, 
	PRIMARY KEY (jockey_id), 
	UNIQUE (canonical_name)
);
CREATE TABLE IF NOT EXISTS owners (
	owner_id SERIAL NOT NULL, 
	canonical_name VARCHAR(200) NOT NULL, 
	display_name VARCHAR(200) NOT NULL, 
	PRIMARY KEY (owner_id), 
	UNIQUE (canonical_name)
);
CREATE TABLE IF NOT EXISTS racecourses (
	racecourse_id SERIAL NOT NULL, 
	code VARCHAR(20) NOT NULL, 
	name VARCHAR(200), 
	country VARCHAR(10), 
	lat FLOAT, 
	lon FLOAT, 
	PRIMARY KEY (racecourse_id), 
	UNIQUE (code)
);
CREATE TABLE IF NOT EXISTS raw_payloads (
	payload_id SERIAL NOT NULL, 
	source_key VARCHAR(40) NOT NULL, 
	endpoint VARCHAR(80) NOT NULL, 
	url VARCHAR(1000) NOT NULL, 
	params_json JSON, 
	fetched_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	http_status INTEGER NOT NULL, 
	sha256 VARCHAR(64) NOT NULL, 
	size_bytes INTEGER NOT NULL, 
	storage_uri VARCHAR(1000) NOT NULL, 
	PRIMARY KEY (payload_id)
);
CREATE INDEX IF NOT EXISTS ix_raw_payloads_url_fetched ON raw_payloads (url, fetched_at);
CREATE TABLE IF NOT EXISTS research_documents (
	doc_id SERIAL NOT NULL, 
	url VARCHAR(1000) NOT NULL, 
	source_name VARCHAR(200), 
	published_at TIMESTAMP WITH TIME ZONE, 
	collected_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	reliability VARCHAR(20), 
	expires_at TIMESTAMP WITH TIME ZONE, 
	content_sha256 VARCHAR(64), 
	summary TEXT, 
	structured_json JSON, 
	PRIMARY KEY (doc_id)
);
CREATE TABLE IF NOT EXISTS stored_files (
	key VARCHAR(300) NOT NULL, 
	content BYTEA NOT NULL, 
	sha256 VARCHAR(64) NOT NULL, 
	size_bytes INTEGER NOT NULL, 
	content_type VARCHAR(80) NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (key)
);
CREATE TABLE IF NOT EXISTS trainers (
	trainer_id SERIAL NOT NULL, 
	canonical_name VARCHAR(200) NOT NULL, 
	display_name VARCHAR(200) NOT NULL, 
	PRIMARY KEY (trainer_id), 
	UNIQUE (canonical_name)
);
CREATE TABLE IF NOT EXISTS model_versions (
	model_id SERIAL NOT NULL, 
	family VARCHAR(60) NOT NULL, 
	version VARCHAR(40) NOT NULL, 
	feature_set_id INTEGER, 
	train_start DATE, 
	train_end DATE, 
	hyperparams_json JSON, 
	artifact_uri VARCHAR(1000), 
	artifact_sha256 VARCHAR(64), 
	git_commit VARCHAR(40), 
	status VARCHAR(20) NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (model_id), 
	CONSTRAINT uq_model_version UNIQUE (family, version), 
	FOREIGN KEY(feature_set_id) REFERENCES feature_sets (feature_set_id)
);
CREATE TABLE IF NOT EXISTS races (
	race_id SERIAL NOT NULL, 
	race_date DATE NOT NULL, 
	reunion_num INTEGER NOT NULL, 
	course_num INTEGER NOT NULL, 
	racecourse_id INTEGER, 
	country VARCHAR(10), 
	label VARCHAR(300), 
	discipline VARCHAR(40), 
	discipline_raw VARCHAR(40), 
	specialite_raw VARCHAR(40), 
	distance_m INTEGER, 
	corde VARCHAR(40), 
	prize_eur FLOAT, 
	declared_runners INTEGER, 
	sex_condition VARCHAR(40), 
	conditions_text TEXT, 
	scheduled_start_utc TIMESTAMP WITH TIME ZONE, 
	status VARCHAR(40), 
	result_final BOOLEAN, 
	source_key VARCHAR(40) NOT NULL, 
	first_observed_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	last_observed_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	payload_id INTEGER, 
	PRIMARY KEY (race_id), 
	CONSTRAINT uq_race_key UNIQUE (race_date, reunion_num, course_num), 
	FOREIGN KEY(racecourse_id) REFERENCES racecourses (racecourse_id), 
	FOREIGN KEY(payload_id) REFERENCES raw_payloads (payload_id)
);
CREATE INDEX IF NOT EXISTS ix_races_date ON races (race_date);
CREATE TABLE IF NOT EXISTS weather_conditions (
	weather_id SERIAL NOT NULL, 
	racecourse_id INTEGER NOT NULL, 
	valid_time_utc TIMESTAMP WITH TIME ZONE NOT NULL, 
	kind VARCHAR(20) NOT NULL, 
	temp_c FLOAT, 
	precip_mm FLOAT, 
	wind_ms FLOAT, 
	issued_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	observed_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (weather_id), 
	FOREIGN KEY(racecourse_id) REFERENCES racecourses (racecourse_id)
);
CREATE TABLE IF NOT EXISTS model_metrics (
	metric_id SERIAL NOT NULL, 
	model_id INTEGER NOT NULL, 
	eval_protocol VARCHAR(60) NOT NULL, 
	period_start DATE, 
	period_end DATE, 
	segment VARCHAR(100), 
	metric VARCHAR(60) NOT NULL, 
	value FLOAT, 
	ci_low FLOAT, 
	ci_high FLOAT, 
	n_races INTEGER, 
	n_entries INTEGER, 
	PRIMARY KEY (metric_id), 
	FOREIGN KEY(model_id) REFERENCES model_versions (model_id)
);
CREATE TABLE IF NOT EXISTS odds_snapshots (
	odds_id SERIAL NOT NULL, 
	race_id INTEGER NOT NULL, 
	num_pmu INTEGER NOT NULL, 
	bet_type VARCHAR(40) NOT NULL, 
	pool VARCHAR(20) NOT NULL, 
	odds_kind VARCHAR(20) NOT NULL, 
	odds FLOAT NOT NULL, 
	published_at TIMESTAMP WITH TIME ZONE, 
	observed_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	payload_id INTEGER, 
	PRIMARY KEY (odds_id), 
	FOREIGN KEY(race_id) REFERENCES races (race_id), 
	FOREIGN KEY(payload_id) REFERENCES raw_payloads (payload_id)
);
CREATE INDEX IF NOT EXISTS ix_odds_race_obs ON odds_snapshots (race_id, observed_at);
CREATE TABLE IF NOT EXISTS payouts (
	payout_id SERIAL NOT NULL, 
	race_id INTEGER NOT NULL, 
	bet_type VARCHAR(60) NOT NULL, 
	pool VARCHAR(20) NOT NULL, 
	combination VARCHAR(100), 
	payout_per_eur FLOAT, 
	observed_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	payload_id INTEGER, 
	PRIMARY KEY (payout_id), 
	CONSTRAINT uq_payout UNIQUE (race_id, pool, bet_type, combination), 
	FOREIGN KEY(race_id) REFERENCES races (race_id), 
	FOREIGN KEY(payload_id) REFERENCES raw_payloads (payload_id)
);
CREATE TABLE IF NOT EXISTS predictions (
	prediction_id SERIAL NOT NULL, 
	race_id INTEGER NOT NULL, 
	model_id INTEGER NOT NULL, 
	decision_time TIMESTAMP WITH TIME ZONE NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	data_status VARCHAR(30) NOT NULL, 
	input_snapshot_hash VARCHAR(64) NOT NULL, 
	PRIMARY KEY (prediction_id), 
	FOREIGN KEY(race_id) REFERENCES races (race_id), 
	FOREIGN KEY(model_id) REFERENCES model_versions (model_id)
);
CREATE TABLE IF NOT EXISTS race_entries (
	entry_id SERIAL NOT NULL, 
	race_id INTEGER NOT NULL, 
	num_pmu INTEGER NOT NULL, 
	horse_id INTEGER NOT NULL, 
	jockey_id INTEGER, 
	trainer_id INTEGER, 
	owner_id INTEGER, 
	age INTEGER, 
	sex VARCHAR(20), 
	draw INTEGER, 
	weight_kg FLOAT, 
	weight_carried_kg FLOAT, 
	handicap_distance_m INTEGER, 
	blinkers VARCHAR(40), 
	shoes VARCHAR(60), 
	status VARCHAR(30), 
	form_string VARCHAR(200), 
	career_starts INTEGER, 
	career_wins INTEGER, 
	career_places INTEGER, 
	career_earnings FLOAT, 
	observed_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	observed_after_start BOOLEAN NOT NULL, 
	content_hash VARCHAR(64) NOT NULL, 
	payload_id INTEGER, 
	PRIMARY KEY (entry_id), 
	FOREIGN KEY(race_id) REFERENCES races (race_id), 
	FOREIGN KEY(horse_id) REFERENCES horses (horse_id), 
	FOREIGN KEY(jockey_id) REFERENCES jockeys (jockey_id), 
	FOREIGN KEY(trainer_id) REFERENCES trainers (trainer_id), 
	FOREIGN KEY(owner_id) REFERENCES owners (owner_id), 
	FOREIGN KEY(payload_id) REFERENCES raw_payloads (payload_id)
);
CREATE INDEX IF NOT EXISTS ix_entries_horse ON race_entries (horse_id);
CREATE INDEX IF NOT EXISTS ix_entries_race_num_obs ON race_entries (race_id, num_pmu, observed_at);
CREATE TABLE IF NOT EXISTS race_results (
	result_id SERIAL NOT NULL, 
	race_id INTEGER NOT NULL, 
	num_pmu INTEGER NOT NULL, 
	horse_id INTEGER NOT NULL, 
	finish_position INTEGER, 
	incident VARCHAR(60), 
	time_s FLOAT, 
	reduction_km_s FLOAT, 
	observed_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	payload_id INTEGER, 
	PRIMARY KEY (result_id), 
	CONSTRAINT uq_result UNIQUE (race_id, num_pmu), 
	FOREIGN KEY(race_id) REFERENCES races (race_id), 
	FOREIGN KEY(horse_id) REFERENCES horses (horse_id), 
	FOREIGN KEY(payload_id) REFERENCES raw_payloads (payload_id)
);
CREATE INDEX IF NOT EXISTS ix_results_horse ON race_results (horse_id);
CREATE TABLE IF NOT EXISTS racecourse_conditions (
	condition_id SERIAL NOT NULL, 
	race_id INTEGER NOT NULL, 
	going_label VARCHAR(80), 
	penetrometer VARCHAR(20), 
	observed_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (condition_id), 
	FOREIGN KEY(race_id) REFERENCES races (race_id)
);
CREATE TABLE IF NOT EXISTS prediction_entries (
	prediction_id INTEGER NOT NULL, 
	num_pmu INTEGER NOT NULL, 
	p_win FLOAT, 
	p_place FLOAT, 
	p_win_lo FLOAT, 
	p_win_hi FLOAT, 
	rank INTEGER, 
	model_disagreement FLOAT, 
	ood_score FLOAT, 
	mkt_prob FLOAT, 
	mkt_odds FLOAT, 
	PRIMARY KEY (prediction_id, num_pmu), 
	FOREIGN KEY(prediction_id) REFERENCES predictions (prediction_id)
);
CREATE TABLE IF NOT EXISTS prediction_explanations (
	explanation_id SERIAL NOT NULL, 
	prediction_id INTEGER NOT NULL, 
	num_pmu INTEGER NOT NULL, 
	factor VARCHAR(100) NOT NULL, 
	contribution FLOAT, 
	direction VARCHAR(10), 
	text TEXT, 
	PRIMARY KEY (explanation_id), 
	FOREIGN KEY(prediction_id) REFERENCES predictions (prediction_id)
);
