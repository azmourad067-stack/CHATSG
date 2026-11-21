-- Prédictions du jour précalculées par la collecte (toutes les 15 min) : l'interface les lit
-- au lieu de tout recalculer. Écrasées à chaque calcul, purgées après quelques jours.
CREATE TABLE IF NOT EXISTS day_prediction_races (
    race_id INTEGER PRIMARY KEY REFERENCES races (race_id),
    race_date DATE NOT NULL,
    data_key VARCHAR(300) NOT NULL,
    primary_model VARCHAR(40),
    message VARCHAR(300),
    computed_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_day_prediction_races_date ON day_prediction_races (race_date);

CREATE TABLE IF NOT EXISTS day_prediction_runners (
    race_id INTEGER NOT NULL REFERENCES day_prediction_races (race_id) ON DELETE CASCADE,
    num_pmu INTEGER NOT NULL,
    horse_id INTEGER,
    p_win DOUBLE PRECISION,
    mkt_odds DOUBLE PRECISION,
    mkt_prob DOUBLE PRECISION,
    rank DOUBLE PRECISION,
    PRIMARY KEY (race_id, num_pmu)
);

-- même régime que les autres tables : RLS, rien pour l'API publique, lecture pour hi_app
do $$
declare t text;
begin
  foreach t in array array['day_prediction_races', 'day_prediction_runners']
  loop
    execute format('alter table public.%I enable row level security', t);
    execute format('revoke all on table public.%I from anon, authenticated', t);
    execute format('grant select on table public.%I to hi_app', t);
    execute format('drop policy if exists hi_app_read on public.%I', t);
    execute format('create policy hi_app_read on public.%I for select to hi_app using (true)', t);
  end loop;
end $$;
