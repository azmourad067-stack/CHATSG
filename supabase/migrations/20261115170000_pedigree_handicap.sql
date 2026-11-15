-- Origines du cheval (père, mère, père de la mère) et valeur handicap officielle (galop),
-- fournies par l'API PMU (participants). Renseignées par la collecte et par
-- scripts/backfill_pedigree.py pour l'historique.
ALTER TABLE horses ADD COLUMN IF NOT EXISTS sire VARCHAR(200);
ALTER TABLE horses ADD COLUMN IF NOT EXISTS dam VARCHAR(200);
ALTER TABLE horses ADD COLUMN IF NOT EXISTS dam_sire VARCHAR(200);
ALTER TABLE race_entries ADD COLUMN IF NOT EXISTS handicap_value FLOAT;
