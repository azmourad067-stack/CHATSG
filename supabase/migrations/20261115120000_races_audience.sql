-- Audience de la réunion PMU : NATIONAL (programme national de pmu.fr) ou REGIONAL / LOCAL
-- (réunions régionales, souvent sans paris en ligne). Renseignée par la collecte.
ALTER TABLE races ADD COLUMN IF NOT EXISTS audience VARCHAR(20);
