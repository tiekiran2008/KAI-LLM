-- KAI-LLM Phase 2: long-term memory table with pgvector + RLS.
-- Run this in the Supabase SQL editor (requires the Vector/pgvector extension).

create extension if not exists vector;

-- Memory table: one row per approved user memory.
-- embedding uses vector(384) to match all-MiniLM-L6-v2.
create table if not exists public.memories (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references auth.users (id) on delete cascade,
    content text not null,
    embedding vector(384),
    created_at timestamptz not null default now()
);

create index if not exists memories_user_id_created_at_idx
    on public.memories (user_id, created_at desc);

-- HNSW index for fast approximate cosine search (pgvector >= 0.5).
create index if not exists memories_embedding_hnsw_idx
    on public.memories using hnsw (embedding vector_cosine_ops);

alter table public.memories enable row level security;

drop policy if exists "memories_select_own" on public.memories;
create policy "memories_select_own"
    on public.memories for select
    using (auth.uid() = user_id);

drop policy if exists "memories_insert_own" on public.memories;
create policy "memories_insert_own"
    on public.memories for insert
    with check (auth.uid() = user_id);

drop policy if exists "memories_delete_own" on public.memories;
create policy "memories_delete_own"
    on public.memories for delete
    using (auth.uid() = user_id);

-- Cosine similarity search, limited to the calling user's memories.
-- SECURITY INVOKER: RLS applies; with the anon key a user can only see
-- their own rows. match_count is hard-capped at 20.
create or replace function public.match_memories(
    query_user_id uuid,
    query_embedding vector(384),
    match_count integer default 5
)
returns table (
    id uuid,
    content text,
    created_at timestamptz,
    similarity float
)
language sql
stable
as $$
    select
        m.id,
        m.content,
        m.created_at,
        1 - (m.embedding <=> query_embedding) as similarity
    from public.memories m
    where m.user_id = query_user_id
      and m.embedding is not null
    order by m.embedding <=> query_embedding
    limit least(greatest(match_count, 1), 20);
$$;

grant execute on function public.match_memories(uuid, vector, integer) to authenticated;
revoke execute on function public.match_memories(uuid, vector, integer) from public, anon;
