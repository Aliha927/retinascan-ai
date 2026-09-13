-- Run this once in Supabase: Dashboard -> SQL Editor -> New Query -> paste -> Run

create table if not exists scans (
    id bigint generated always as identity primary key,
    user_email text,
    patient_name text,
    prediction text,
    confidence real,
    clinical_report text,
    patient_report text,
    urdu_report text,
    timestamp text
);

-- Enable Row Level Security (recommended even for a hackathon demo)
alter table scans enable row level security;

-- Allow the backend (using the anon key) to insert and read rows.
-- This is permissive for demo speed — in a real product you'd restrict
-- reads/writes to the authenticated user's own rows using auth.uid().
create policy "Allow insert for all" on scans
    for insert
    with check (true);

create policy "Allow select for all" on scans
    for select
    using (true);
