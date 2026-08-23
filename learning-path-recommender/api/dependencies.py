"""Shared FastAPI dependencies (DB session, auth, rate limiting).

Real DB session wiring lands in Phase 3, once the ORM layer over the
star schema is in place.
"""


def get_db():
    raise NotImplementedError("Implemented in Phase 3 - API integration")
