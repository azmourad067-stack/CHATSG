-- Alertes de pari envoyées par la collecte (SMS) : une seule par règle, course et cheval.
CREATE TABLE IF NOT EXISTS alerts_sent (
    rule VARCHAR(40) NOT NULL,
    race_id INTEGER NOT NULL REFERENCES races (race_id),
    num_pmu INTEGER NOT NULL,
    channel VARCHAR(20) NOT NULL,
    sent_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (rule, race_id, num_pmu)
);

-- même régime que les autres tables ; la collecte (hi_writer) écrit, l'interface lit
do $$
declare t text;
begin
  foreach t in array array['alerts_sent']
  loop
    execute format('alter table public.%I enable row level security', t);
    execute format('revoke all on table public.%I from anon, authenticated', t);
    execute format('grant select on table public.%I to hi_app', t);
    execute format('create policy hi_app_read on public.%I for select to hi_app using (true)', t);
    execute format('grant select, insert, update, delete on table public.%I to hi_writer', t);
    execute format('create policy hi_writer_all on public.%I for all to hi_writer '
                   'using (true) with check (true)', t);
  end loop;
end $$;
