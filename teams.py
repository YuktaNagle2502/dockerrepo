from fastapi import APIRouter, Depends, Query, Path, Request, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import cast, String, or_, func
from uuid import UUID
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
from decimal import Decimal
import pytz
from app.schemas.team import TeamCreate, TeamUpdate
from app.schemas.teams_llm_config import TeamsLLMConfigRequest
from app.models.models import Team, TeamRoleUserMapping,SessionPolicy,TokenQuotaCounter,TeamTokenQuotaPolicy, User, UserRole, UserRoleMapping, BudgetAlert, TeamsLLMConfig, BudgetAlertRecipient
from app.core.response import success_response, error_response
from app.utils.pagination import paginate_offset
from app.database import get_db
from app.middleware.auth_middleware import role_guard
from app.services.admin_service import get_admin_teams,set_team_quota,update_team_cap,get_team_owner_teams
from typing import List
from fastapi import Request
from app.utils.kms_util import encrypt_text, decrypt_text
from app.core.config import settings
import json
import calendar

team_router = APIRouter(
    prefix="/api/v1/teams",
    tags=["Teams"]
)

#---------GLOBAL VARIABLE--------

TEAM_NOT_FOUND_MESSAGE = "Team not found"

IST = pytz.timezone('Asia/Kolkata')

#--------------------------------

@team_router.get("/token-summary")
def get_team_token_quota_summary(
    team_id: UUID = Query(..., description="UUID of the team"),
    db: Session = Depends(get_db)
):
    try:
        # 1. Total Cap
        total_cap = db.query(func.coalesce(func.sum(TeamTokenQuotaPolicy.token_cap), 0)) \
            .filter(TeamTokenQuotaPolicy.team_id == team_id) \
            .scalar()

        # 2. Total Consumed
        total_consumed = db.query(func.coalesce(func.sum(TokenQuotaCounter.tokens_consumed), 0)) \
            .filter(TokenQuotaCounter.team_id == team_id) \
            .scalar()
        # 3. Cost Cap
        total_cost_cap = db.query(
            func.coalesce(func.sum(TeamTokenQuotaPolicy.cost_cap), 0.0)
        ).filter(TeamTokenQuotaPolicy.team_id == team_id).scalar()

        # 4. Cost Consumed
        total_cost_consumed = db.query(
            func.coalesce(func.sum(TokenQuotaCounter.cost_consumed), 0.0)
        ).filter(TokenQuotaCounter.team_id == team_id).scalar()

        # 5. Threshold Percentages (get all active alerts for this team)
        threshold_percentages = db.query(BudgetAlert.threshold_percentage) \
            .filter(
                BudgetAlert.team_id == team_id,
                BudgetAlert.status == 'active'
            ) \
            .order_by(BudgetAlert.threshold_percentage) \
            .all()
        
        # Extract percentages from tuples and convert to float
        threshold_percentages_list = [float(tp[0]) for tp in threshold_percentages]

        result = {
            "team_id": team_id,
            "total_cap": total_cap,
            "total_consumed": total_consumed,
            "remaining": total_cap - total_consumed if total_cap is not None else 0,
            "cost_cap": float(total_cost_cap),
            "cost_consumed": float(total_cost_consumed),
            "cost_remaining": float(total_cost_cap - total_cost_consumed),
            "threshold_percentages": threshold_percentages_list
        }
        
        return success_response(data=result, message="Team token summary fetched successfully")

    except Exception as e:
        return error_response(
            message="Failed to fetch token summary",
            error_key="INTERNAL_ERROR",
            details=str(e)
        )

        
@team_router.get("/user-teams")
def get_user_teams(
    user_id: UUID = Query(..., description="UUID of the user"),
    db: Session = Depends(get_db),
):
    """
    Get all teams of a user by email.
    """
    try:
        # Get all team mappings for this user
        team_mappings = (
            db.query(TeamRoleUserMapping)
            .join(Team)
            .filter(
                TeamRoleUserMapping.user_id == user_id,
                TeamRoleUserMapping.is_deleted == False,
                Team.is_deleted == False
            )
            .all()
        )
        teams_list = []
        if not team_mappings:
            return success_response(teams_list, message="Teams fetched successfully")

        # Get TEAM_OWNER role once
        team_owner_role = db.query(UserRole).filter(
            UserRole.role_name.ilike("TEAM_OWNER")
        ).first()

        # Extract unique teams and resolve owner
        seen_team_ids = set()

        for mapping in team_mappings:
            team = mapping.team
            if team and team.id not in seen_team_ids:
                owner_id = None
                if team_owner_role:
                    owner_mapping = db.query(UserRoleMapping).filter(
                        UserRoleMapping.scope_type == "TEAM",
                        UserRoleMapping.scope_id == team.id,
                        UserRoleMapping.user_role_id == team_owner_role.id
                    ).first()
                    if owner_mapping:
                        owner_id = str(owner_mapping.user_id)

                teams_list.append({
                    "team_id": str(team.id),
                    "team_name": team.name,
                    "organization_id": str(team.organization_id),
                    "team_owner_id": owner_id  
                })
                seen_team_ids.add(team.id)

        return success_response(teams_list, message="Teams fetched successfully")

    except Exception as e:
        return error_response(
            message="Failed to fetch user teams",
            error_key="INTERNAL_ERROR",
            details=str(e)
        )

def last_day_of_month(dt: datetime = None) -> datetime:
    if dt is None:
        dt = datetime.now(IST)
    # Localize or convert to IST first
    if dt.tzinfo is None:
        dt = IST.localize(dt)
    else:
        dt = dt.astimezone(IST)
        
    last_day = calendar.monthrange(dt.year, dt.month)[1]
    return IST.localize(datetime(dt.year, dt.month, last_day, 23, 59, 59))

@team_router.post(
    "",
    response_model=None,
    dependencies=[Depends(role_guard(["SUPER_ADMIN", "ADMIN"]))]
)
async def create_team(request: Request, payload: TeamCreate, db: Session = Depends(get_db)):
    try:
        # 1️⃣ Get current user info and roles
        current_user_email = request.state.user_email
        highest_role = request.state.user_highest_role  
        target_payload = payload.dict()
        target_payload.pop("token_balance", None)  # remove if present
        target_payload.pop("cost_cap", None)
        target_payload.pop("auto_renew", None)
        target_payload.pop("effective_to", None)

        # 2️⃣ ORG validation
        org_id_from_payload = target_payload.get("organization_id")
        if highest_role == "SUPER_ADMIN":
            if not org_id_from_payload:
                return error_response(
                    message="organization_id is required for SUPER_ADMIN",
                    error_key="MISSING_PARAMETER",
                    status_code=400
                )
        elif highest_role == "ADMIN":
            if org_id_from_payload:
                return error_response(
                    message="Only SUPER_ADMIN can provide organization_id explicitly",
                    error_key="FORBIDDEN",
                    status_code=403
                )
            token_org_id = request.state.organization_id
            if not token_org_id:
                return error_response(
                    message="Organization ID not found in token for ADMIN",
                    error_key="MISSING_PARAMETER",
                    status_code=400
                )
            target_payload["organization_id"] = token_org_id

        target_payload["created_by"] = current_user_email
        target_payload["updated_by"] = current_user_email

        # 3️⃣ CREATE TEAM
        new_team = Team(**target_payload)
        db.add(new_team)
        db.flush()

        # ---------------- SESSION POLICY ----------------

        now_ist = datetime.now(IST)

        session_policy = SessionPolicy(
            team_id=new_team.id,
            max_session_age_days=30,
            max_messages_per_session=1000,
            conversation_retention_days=30,
            effective_from=now_ist.replace(tzinfo=None),
            effective_to=IST.localize(datetime(2026, 8, 28, 16, 5)).replace(tzinfo=None),
            created_by="system",
            updated_by="system"
        )
        db.add(session_policy)
        db.flush()

        now = datetime.now(IST)

        if payload.auto_renew is False and payload.effective_to is None:
            return error_response(
                message="effective_to is required when auto_renew is false",
                error_key="VALIDATION_ERROR",
                status_code=400
            )

        if payload.auto_renew:
            effective_from = now
            effective_to = last_day_of_month(now)
        else:
            effective_from = now
            effective_to = payload.effective_to 

        # ---------------- TEAM TOKEN QUOTA POLICY ----------------
        team_quota_policy = TeamTokenQuotaPolicy(
            team_id=new_team.id,
            token_cap=payload.token_balance,
            cost_cap=payload.cost_cap,
            auto_renew=payload.auto_renew,
            effective_from=effective_from.replace(tzinfo=None),
            effective_to=effective_to.replace(tzinfo=None),
            created_by="system",
            updated_by="system"
        )
        db.add(team_quota_policy)
        db.flush()

        # ---------------- TOKEN QUOTA COUNTER ----------------
        token_quota_counter = TokenQuotaCounter(
            team_id=new_team.id,
            team_token_quota_policy_id=team_quota_policy.id,
            tokens_consumed=0,
            cost_consumed=0,
            created_by="system"
        )
        db.add(token_quota_counter)
        db.flush()

        # ---------------- USER ROLE MAPPING ----------------
        team_owner_id = payload.team_owner_id
        if not team_owner_id:
            return error_response(
                message="Team owner ID is required",
                error_key="MISSING_PARAMETER",
                status_code=400
            )

        team_owner_role = db.query(UserRole).filter(UserRole.role_name.ilike("TEAM_OWNER")).first()
        if not team_owner_role:
            return error_response(
                message="TEAM_OWNER role not found in UserRole table",
                error_key="MISSING_ROLE",
                status_code=400
            )

        role_mapping = UserRoleMapping(
            user_id=team_owner_id,
            user_role_id=team_owner_role.id,
            scope_type="TEAM",
            scope_id=new_team.id,
            created_by=current_user_email,
            updated_by=current_user_email,
        )
        db.add(role_mapping)

        # ---------------- COMMIT ----------------
        db.commit()
        db.refresh(new_team)

        # sync cap into Redis
        await set_team_quota(str(new_team.id), team_quota_policy.cost_cap, 0)
        return success_response(
            data={
                "team": new_team,
                "session_policy": session_policy,
                "team_token_quota_policy": team_quota_policy,
                "token_quota_counter": token_quota_counter,
                "role_mapping": role_mapping
            },
            message="Team added successfully",
            status_code=201
        )

    except Exception as e:
        db.rollback()
        return error_response(
            message=f"Failed to add new team: {str(e)}",
            error_key="DB_ERROR"
        )


# ---------------- LIST ----------------
def _get_team_roles_count(team_id: UUID, db: Session) -> int:
    """Get count of distinct roles for a team."""
    return db.query(func.count(func.distinct(TeamRoleUserMapping.role_id))).filter(
        TeamRoleUserMapping.team_id == team_id,
        TeamRoleUserMapping.is_deleted == False
    ).scalar() or 0


def _get_team_users_count(team_id: UUID, db: Session) -> int:
    """Get count of distinct users for a team."""
    return db.query(func.count(func.distinct(TeamRoleUserMapping.user_id))).filter(
        TeamRoleUserMapping.team_id == team_id,
        TeamRoleUserMapping.is_deleted == False
    ).scalar() or 0


def _get_team_owner_data(team_id: UUID, db: Session):
    """Fetch team owner details using UserRoleMapping."""
    owner_role = db.query(UserRole).filter(UserRole.role_name.ilike("TEAM_OWNER")).first()
    if not owner_role:
        return None

    owner_mapping = db.query(UserRoleMapping).filter(
        UserRoleMapping.scope_id == team_id,
        UserRoleMapping.scope_type == "TEAM",
        UserRoleMapping.user_role_id == owner_role.id
    ).first()
    
    if not owner_mapping:
        return None

    owner = db.query(User).filter(User.id == owner_mapping.user_id).first()
    if not owner:
        return None

    return {
        "id": str(owner.id),
        "first_name": owner.first_name,
        "last_name": owner.last_name,
        "email": owner.email,
        "organization_id": str(owner.organization_id) if owner.organization_id else None,
        "created_by": owner.created_by,
        "updated_by": owner.updated_by,
        "created_at": owner.created_at.isoformat() if owner.created_at else None,
        "updated_at": owner.updated_at.isoformat() if owner.updated_at else None
    }


def _get_token_quota_data(team_id: UUID, db: Session):
    """Fetch token quota and consumption data for a team."""
    token_quota = db.query(TeamTokenQuotaPolicy).filter(
        TeamTokenQuotaPolicy.team_id == team_id
    ).first()
    
    total_tokens = token_quota.token_cap if token_quota and token_quota.token_cap is not None else 0
    cost_cap = token_quota.cost_cap if token_quota and token_quota.cost_cap is not None else 0
    
    consumed_tokens = db.query(func.coalesce(func.sum(TokenQuotaCounter.tokens_consumed), 0)).filter(
        TokenQuotaCounter.team_id == team_id
    ).scalar() or 0
    
    consumed_cost = db.query(func.coalesce(func.sum(TokenQuotaCounter.cost_consumed), 0)).filter(
        TokenQuotaCounter.team_id == team_id
    ).scalar() or 0
    
    is_allocated = bool(token_quota and token_quota.cost_cap and token_quota.cost_cap > 0)
    
    return {
        "total_tokens": total_tokens,
        "cost_cap": cost_cap,
        "consumed_tokens": consumed_tokens,
        "consumed_cost": consumed_cost,
        "is_allocated": is_allocated
    }


def _build_team_data_dict(team, owner_data, roles_count: int, users_count: int, quota_data: dict):
    """Build team data dictionary with all required fields."""
    return {
        "id": str(team.id),
        "organization_id": str(team.organization_id),
        "name": team.name,
        "team_owner": [owner_data] if owner_data else [],
        "roles_count": roles_count,
        "users_count": users_count,
        "is_allocated": quota_data["is_allocated"],
        "total_tokens": quota_data["total_tokens"],
        "cost_cap": quota_data["cost_cap"],
        "consumed_tokens": quota_data["consumed_tokens"],
        "consumed_cost": quota_data["consumed_cost"],
        "created_by": team.created_by,
        "updated_by": team.updated_by,
        "created_at": team.created_at.isoformat() if team.created_at else None,
        "updated_at": team.updated_at.isoformat() if team.updated_at else None
    }


def _process_team_data(team, db: Session):
    """Process a single team and return its complete data."""
    roles_count = _get_team_roles_count(team.id, db)
    users_count = _get_team_users_count(team.id, db)
    owner_data = _get_team_owner_data(team.id, db)
    quota_data = _get_token_quota_data(team.id, db)
    
    return _build_team_data_dict(team, owner_data, roles_count, users_count, quota_data)


def _apply_search_filters(query, search: str):
    """Apply search filters to team query."""
    if not search:
        return query
    
    filters = []
    for column in Team.__table__.columns:
        if column.name == "is_deleted":
            continue
        filters.append(cast(column, String).ilike(f"%{search}%"))
    
    if filters:
        query = query.filter(or_(*filters))
    
    return query


def _get_superadmin_teams_query(organization_id: UUID, db: Session):
    """Get base query for superadmin teams."""
    return db.query(Team).filter(
        Team.organization_id == organization_id,
        Team.is_deleted == False
    )


def _process_teams_list(items: list, db: Session):
    """Process list of teams and return team data."""
    return [_process_team_data(team, db) for team in items]


def _handle_superadmin_all_teams(organization_id: UUID, db: Session):
    """Handle superadmin request for all teams (size=-1)."""
    query = _get_superadmin_teams_query(organization_id, db)
    items = query.all()
    team_data = _process_teams_list(items, db)
    return {"items": team_data}, items


def _handle_superadmin_paginated_teams(organization_id: UUID, page: int, size: int, search: str, db: Session):
    """Handle superadmin request for paginated teams."""
    query = _get_superadmin_teams_query(organization_id, db)
    query = _apply_search_filters(query, search)
    items, meta = paginate_offset(query, page, size)
    team_data = _process_teams_list(items, db)
    return {"items": team_data, "meta": meta}, items


def _handle_superadmin_request(organization_id: UUID, page: int, size: int, search: str, db: Session):
    """Handle SUPER_ADMIN role team listing."""
    if not organization_id:
        return error_response(
            message="organization_id is required for SUPER_ADMIN",
            error_key="MISSING_PARAMETER",
            status_code=400
        )
    
    if size == -1:
        payload, items = _handle_superadmin_all_teams(organization_id, db)
    else:
        payload, items = _handle_superadmin_paginated_teams(organization_id, page, size, search, db)
    
    if not items:
        return success_response(
            data=payload,
            message="No teams found for the organization"
        )

    return success_response(
        data=payload,
        message="Teams retrieved successfully"
    )


@team_router.get("", response_model=None,dependencies=[Depends(role_guard(["SUPER_ADMIN","ADMIN","TEAM_OWNER"]))])
def list_teams(
    request:Request,
    page: int = Query(1, ge=1, description="Page number (ignored if size=0)"),
    size: int = Query(10, ge=-1, le=100, description="Number of items per page, -1 for all items"),
    search: str | None = Query(None, description="Search across all columns"),
    organization_id: UUID | None = Query(None, description="Filter teams by organization ID (for superadmin)"),
    db: Session = Depends(get_db),
):
    user_id = request.state.user_id
    highest_role = request.state.user_highest_role
    
    try:
        if highest_role == "SUPER_ADMIN":
            return _handle_superadmin_request(organization_id, page, size, search, db)
        elif highest_role == "ADMIN":
            return get_admin_teams(user_id, page, size, search, db)
        elif highest_role == "TEAM_OWNER":
            return get_team_owner_teams(user_id, page, size, search, db)
        else:
            return error_response(
                message="Role not allowed to fetch teams",
                error_key="FORBIDDEN",
                status_code=403
            )
    except Exception as e:
        return error_response(
            message=f"Failed to fetch Teams: {str(e)}",
            error_key="DB_ERROR",
            status_code=500
        )

# ---------------- GET BY ID ----------------
@team_router.get("/{team_id}", response_model=None,dependencies=[Depends(role_guard(["SUPER_ADMIN","ADMIN","TEAM_OWNER"]))])
def get_team(team_id: UUID = Path(...), db: Session = Depends(get_db)):
    try:
        team = db.query(Team).filter(Team.id == team_id, Team.is_deleted == False).first()
        if not team:
            return error_response(
                message=TEAM_NOT_FOUND_MESSAGE,
                error_key="NOT_FOUND",
                status_code=404
            )

        # 🔎 Fetch TEAM_OWNER role
        team_owner_role = db.query(UserRole).filter(
            UserRole.role_name.ilike("TEAM_OWNER")
        ).first()

        owner_id = None
        if team_owner_role:
            mapping = db.query(UserRoleMapping).filter(
                UserRoleMapping.scope_type == "TEAM",
                UserRoleMapping.scope_id == team_id,
                UserRoleMapping.user_role_id == team_owner_role.id
            ).first()
            if mapping:
                owner_id = str(mapping.user_id)

        # Build response
        response_data = {
            "id": str(team.id),
            "organization_id": str(team.organization_id),
            "name": team.name,
            "team_owner_id": owner_id,   
            "is_deleted": team.is_deleted,
            "created_by": team.created_by,
            "updated_by": team.updated_by,
            "created_at": team.created_at.isoformat() if team.created_at else None,
            "updated_at": team.updated_at.isoformat() if team.updated_at else None,
        }

        return success_response(
            data=response_data,
            message="Team retrieved successfully"
        )

    except Exception as e:
        return error_response(
            message=f"Failed to fetch Team: {str(e)}",
            error_key="DB_ERROR"
        )

# ---------------- UPDATE ----------------
@team_router.put("/{team_id}", response_model=None,dependencies=[Depends(role_guard(["SUPER_ADMIN","ADMIN","TEAM_OWNER"]))])
async def update_team(request: Request, team_id: UUID, payload: TeamUpdate, db: Session = Depends(get_db)):
    try:
        current_user_email = request.state.user_email
        team = db.query(Team).filter(Team.id == team_id, Team.is_deleted == False).first()
        if not team:
            return error_response(
                message="Team not found",
                error_key="NOT_FOUND",
                status_code=404
            )
 
        # Extract token_balance from payload before updating team
        update_data = payload.dict(exclude_unset=True)
        token_balance = update_data.pop("token_balance", None)
        cost_cap = update_data.pop("cost_cap", None)
        auto_renew = update_data.pop("auto_renew", None)  # <-- REQUIRED
        effective_to = update_data.pop("effective_to", None)
        new_owner_id = update_data.pop("team_owner_id", None)

        if auto_renew is False and effective_to is None:
            return error_response(
                message="When auto_renew is false, effective_to date is required",
                error_key="VALIDATION_ERROR",
                status_code=400
            )
 
        # Update team basic fields (excluding owner + token stuff)
        for key, value in update_data.items():
            setattr(team, key, value)
        team.updated_at = datetime.now(timezone.utc)
        team.updated_by = current_user_email
 
        # -----------------------------
        # Handle TEAM_OWNER update via UserRoleMapping and Team.team_owner_id
        # -----------------------------
        if new_owner_id:
            print(f"Before update: Team {team_id}, team_owner_id: {team.team_owner_id}")
            team_owner_role = db.query(UserRole).filter(
                UserRole.role_name.ilike("TEAM_OWNER")
            ).first()
            if not team_owner_role:
                return error_response(
                    message="TEAM_OWNER role not found in UserRole table",
                    error_key="DB_ERROR",
                    details="TEAM_OWNER role missing",
                )
                
            # Validate new_owner_id
            new_owner = db.query(User).filter(User.id == new_owner_id).first()
            if not new_owner:
                print(f"Invalid new_owner_id: {new_owner_id}")  # Debug
                return error_response(
                    message="New team owner not found in users table",
                    error_key="NOT_FOUND",
                    status_code=404
                )
                
            # Update Team.team_owner_id
            team.team_owner_id = new_owner_id
            print(f"After setting: Team {team_id}, team_owner_id: {team.team_owner_id}")
 
            # Remove existing TEAM_OWNER mapping for this team
            existing_mapping = db.query(UserRoleMapping).filter(
                UserRoleMapping.scope_type == "TEAM",
                UserRoleMapping.scope_id == team_id,
                UserRoleMapping.user_role_id == team_owner_role.id
            ).first()
 
            if existing_mapping:
                if existing_mapping.user_id != new_owner_id:
                    # Change ownership: update user_id
                    existing_mapping.user_id = new_owner_id
                    existing_mapping.updated_by = current_user_email
                    existing_mapping.updated_at = datetime.now(timezone.utc)
            else:
                # No mapping exists, create a new one
                new_mapping = UserRoleMapping(
                    user_id=new_owner_id,
                    user_role_id=team_owner_role.id,
                    scope_type="TEAM",
                    scope_id=team_id,
                    created_by=current_user_email,
                    updated_by=current_user_email
                )
                db.add(new_mapping)
 
        # -----------------------------
        # Handle token balance + cost cap updates
        # -----------------------------
        now = datetime.now(IST)   # <-- REQUIRED FIX

        team_quota_policy = db.query(TeamTokenQuotaPolicy).filter(
            TeamTokenQuotaPolicy.team_id == team_id
        ).first()
 
        if team_quota_policy:
            if token_balance is not None:
                team_quota_policy.token_cap = token_balance
            if cost_cap is not None:  # ✅ NEW
                team_quota_policy.cost_cap = cost_cap

            # <-- REQUIRED auto_renew handling
            if auto_renew is not None:
                team_quota_policy.auto_renew = auto_renew

                if auto_renew:
                    team_quota_policy.effective_from = now.replace(tzinfo=None)
                    team_quota_policy.effective_to = last_day_of_month(now).replace(tzinfo=None)
                else:
                    if effective_to:
                        team_quota_policy.effective_from = now.replace(tzinfo=None)
                        team_quota_policy.effective_to = effective_to.replace(tzinfo=None)


            team_quota_policy.updated_by = current_user_email
            team_quota_policy.updated_at = datetime.now(timezone.utc)
        else:
            if auto_renew is False and effective_to is None:
                return error_response(
                    message="When auto_renew is false, effective_to date is required",
                    error_key="VALIDATION_ERROR",
                    status_code=400
                )
            team_quota_policy = TeamTokenQuotaPolicy(
                team_id=team_id,
                token_cap=token_balance,
                cost_cap=cost_cap,
                auto_renew=auto_renew if auto_renew is not None else False,
                rate_per_sec=50,
                rate_per_min=5,
                rate_per_day=5,
                rate_per_month=10,
                effective_from=now.replace(tzinfo=None),   # <-- FIXED
                effective_to=(last_day_of_month(now).replace(tzinfo=None) if auto_renew else effective_to.replace(tzinfo=None)),
                created_by=current_user_email,
                updated_by=current_user_email
            )
            db.add(team_quota_policy)
            db.flush()
 
            existing_counter = db.query(TokenQuotaCounter).filter(
                TokenQuotaCounter.team_id == team_id
            ).first()
 
            if not existing_counter:
                token_quota_counter = TokenQuotaCounter(
                    team_id=team_id,
                    team_token_quota_policy_id=team_quota_policy.id,
                    tokens_consumed=0,
                    cost_consumed=0,
                    created_by=current_user_email
                )
                db.add(token_quota_counter)
        db.commit()
        db.refresh(team)
 
        # 🔄 Update Redis quota
        current_token_balance = 0
        if team_quota_policy and team_quota_policy.cost_cap is not None:
            current_token_balance = team_quota_policy.token_cap
            await update_team_cap(str(team.id), team_quota_policy.cost_cap)
 
        # 🔎 Fetch latest TEAM_OWNER mapping for response
        team_owner_role = db.query(UserRole).filter(UserRole.role_name.ilike("TEAM_OWNER")).first()
        owner_mapping = None
        if team_owner_role:
            owner_mapping = db.query(UserRoleMapping).filter(
                UserRoleMapping.scope_type == "TEAM",
                UserRoleMapping.scope_id == team_id,
                UserRoleMapping.user_role_id == team_owner_role.id
            ).first()
 
        # -----------------------------
        # Response
        # -----------------------------
        response_data = {
            "id": str(team.id),
            "organization_id": str(team.organization_id),
            "name": team.name,
            "team_owner_id": str(owner_mapping.user_id) if owner_mapping else None,
            "token_balance": current_token_balance,
            "cost_cap": float(team_quota_policy.cost_cap) if team_quota_policy else 0.0,
            "auto_renew": team_quota_policy.auto_renew if team_quota_policy else None,  # <-- REQUIRED
            "is_deleted": team.is_deleted,
            "created_by": team.created_by,
            "updated_by": team.updated_by,
            "created_at": team.created_at.isoformat() if team.created_at else None,
            "updated_at": team.updated_at.isoformat() if team.updated_at else None
        }
       
        return success_response(
            data=response_data,
            message="Team details updated successfully"
        )
    except Exception as e:
        db.rollback()
        return error_response(
            message=f"Failed to update team details: {str(e)}",
            error_key="DB_ERROR"
        )

# ---------------- DELETE (SOFT) ----------------
@team_router.delete(
    "/{team_id}",
    response_model=None,
    dependencies=[Depends(role_guard(["SUPER_ADMIN", "ADMIN", "TEAM_OWNER"]))],
)
def delete_team(team_id: UUID, db: Session = Depends(get_db)):
    try:
        team = (
            db.query(Team)
            .filter(Team.id == team_id, Team.is_deleted == False)
            .first()
        )
        if not team:
            return error_response(
                message=TEAM_NOT_FOUND_MESSAGE,
                error_key="NOT_FOUND",
                status_code=404,
            )

        # Soft delete the team
        team.is_deleted = True
        team.updated_at = datetime.now(timezone.utc)

        # Hard delete all TeamRoleUserMappings for this team
        db.query(TeamRoleUserMapping).filter(
            TeamRoleUserMapping.team_id == team_id
        ).delete(synchronize_session=False)

        # Hard delete all UserRoleMappings scoped to this team
        db.query(UserRoleMapping).filter(
            UserRoleMapping.scope_type == "TEAM",
            UserRoleMapping.scope_id == team_id,
        ).delete(synchronize_session=False)

        # Hard delete all BudgetAlerts for this team
        db.query(BudgetAlert).filter(
            BudgetAlert.team_id == team_id
        ).delete(synchronize_session=False)

        # Hard delete all SessionPolicy records for this team
        db.query(SessionPolicy).filter(
            SessionPolicy.team_id == team_id
        ).delete(synchronize_session=False)

        # Hard delete all TeamTokenQuotaPolicy records for this team
        db.query(TeamTokenQuotaPolicy).filter(
            TeamTokenQuotaPolicy.team_id == team_id
        ).delete(synchronize_session=False)

        # Hard delete all TokenQuotaCounter records for this team
        db.query(TokenQuotaCounter).filter(
            TokenQuotaCounter.team_id == team_id
        ).delete(synchronize_session=False)

        db.query(TeamsLLMConfig).filter(
            TeamsLLMConfig.team_id == team_id
        ).delete(synchronize_session=False)

        db.commit()
        return success_response(
            data={"id": str(team.id)},
            message="Team deleted and related mappings deleted successfully",
        )
    except Exception as e:
        db.rollback()
        return error_response(
            message="Failed to delete team",
            error_key="DB_ERROR",
            details=str(e),
        )

@team_router.get("/{team_id}/users",dependencies=[Depends(role_guard(["SUPER_ADMIN","ADMIN","TEAM_OWNER"]))])
def get_team_users_emails(
    team_id: UUID = Path(..., description="Team ID to get user emails for"),
    db: Session = Depends(get_db),
):
    """
    Get unique user emails with first name and last name for users present in TeamRoleUserMapping for a specific team.
    """
    try:
        # Query to get unique users with their emails from TeamRoleUserMapping
        users = (
            db.query(User.email, User.first_name, User.last_name)
            .join(TeamRoleUserMapping, User.id == TeamRoleUserMapping.user_id)
            .filter(
                TeamRoleUserMapping.team_id == team_id,
                TeamRoleUserMapping.is_deleted == False
            )
            .distinct()
            .order_by(User.first_name, User.last_name)
            .all()
        )

        # Format the results
        results = [
            {
                "email": user.email,
                "first_name": user.first_name,
                "last_name": user.last_name
            }
            for user in users
        ]

        if not results:
            # Verify if team exists
            team_exists = db.query(Team).filter(
                Team.id == team_id, 
                Team.is_deleted == False
            ).first()
            
            if not team_exists:
                return error_response(
                    message=TEAM_NOT_FOUND_MESSAGE,
                    error_key="TEAM_NOT_FOUND",
                    status_code=404
                )
            
            return success_response(
                data=[],
                message="No users found in this team"
            )

        return success_response(
            data=results,
            message=f"Team user emails fetched successfully. Found {len(results)} unique user(s)."
        )

    except Exception as e:
        return error_response(
            message="Failed to fetch team user emails",
            details=str(e),
            error_key="FETCH_FAILED",
            status_code=500
        )

def _validate_threshold_percentages(threshold_percentages: List[float]):
    """Validate and return unique threshold percentages."""
    if not threshold_percentages:
        return None, error_response(
            message="At least one threshold percentage is required",
            error_key="MISSING_PARAMETER",
            status_code=400
        )
    
    unique_percentages = list(set(threshold_percentages))
    invalid = [p for p in unique_percentages if p <= 0 or p > 100]
    if invalid:
        return None, error_response(
            message=f"Invalid threshold percentages: {invalid}. Must be between 0 and 100",
            error_key="INVALID_PARAMETER",
            status_code=400
        )
    
    return unique_percentages, None


def _validate_team_and_cost_cap(team_id: UUID, db: Session):
    """Validate team exists and has cost cap configured."""
    team = db.query(Team).filter(Team.id == team_id, Team.is_deleted == False).first()
    if not team:
        return None, None, error_response(
            message=TEAM_NOT_FOUND_MESSAGE,
            error_key="TEAM_NOT_FOUND",
            status_code=404
        )

    total_cost_cap = db.query(func.coalesce(func.sum(TeamTokenQuotaPolicy.cost_cap), 0)) \
        .filter(TeamTokenQuotaPolicy.team_id == team_id).scalar() or 0

    if total_cost_cap == 0:
        return None, None, error_response(
            message="Team has no cost cap configured",
            error_key="NO_COST_CAP",
            status_code=400
        )
    
    return team, total_cost_cap, None


def _get_existing_alerts_map(team_id: UUID, db: Session):
    """Get existing budget alerts as a threshold percentage map."""
    existing_alerts = db.query(BudgetAlert).filter(
        BudgetAlert.team_id == team_id,
    ).all()
    
    return {float(alert.threshold_percentage): alert for alert in existing_alerts}


def _process_alert_deletions(existing_thresholds: dict, requested_thresholds: set, db: Session):
    """Delete alerts not in requested thresholds and return results."""
    deleted_alerts = []
    skipped_alerts = []
    
    for existing_threshold in list(existing_thresholds.keys()):
        alert = existing_thresholds[existing_threshold]
        if existing_threshold not in requested_thresholds:
            if not alert.notified:
                deleted_alerts.append({
                    "threshold_percentage": float(existing_threshold),
                    "alert_name": alert.alert_name
                })
                db.delete(alert)
                existing_thresholds.pop(existing_threshold, None)
            else:
                skipped_alerts.append({
                    "threshold_percentage": float(existing_threshold),
                    "threshold_amount": alert.threshold_amount,
                    "alert_name": alert.alert_name,
                    "reason": "already_notified — retained"
                })
    
    return deleted_alerts, skipped_alerts


def _handle_existing_alert(existing_alert, threshold_percentage: float, threshold_amount: float):
    """Handle logic for existing alert and return skip reason."""
    if existing_alert.notified:
        return {
            "threshold_percentage": float(threshold_percentage),
            "threshold_amount": threshold_amount,
            "alert_name": existing_alert.alert_name,
            "reason": "already_notified — will not recreate"
        }
    else:
        return {
            "threshold_percentage": float(threshold_percentage),
            "threshold_amount": threshold_amount,
            "alert_name": existing_alert.alert_name,
            "reason": "already_exists"
        }


def _create_new_alert(team_id: UUID, team, threshold_percentage: float, threshold_amount: float, db: Session):
    """Create a new budget alert and return alert details."""
    alert_name = f"{team.name}-budget-alert"
    
    new_alert = BudgetAlert(
        team_id=team_id,
        organization_id=team.organization_id,
        alert_name=alert_name,
        threshold_percentage=threshold_percentage,
        threshold_amount=threshold_amount,
        cost_consumed=0,
        notified=False,
        created_by="system",
        updated_by="system"
    )
    db.add(new_alert)
    
    return {
        "threshold_percentage": float(threshold_percentage),
        "threshold_amount": threshold_amount,
        "alert_name": alert_name
    }


def _process_alert_creations(unique_percentages: List[float], existing_thresholds: dict, 
                             team_id: UUID, team, total_cost_cap: float, db: Session):
    """Process alert creation or skipping and return results."""
    created_alerts = []
    skipped_alerts = []
    
    for threshold_percentage in unique_percentages:
        threshold_amount = (threshold_percentage / 100) * float(total_cost_cap)
        existing_alert = existing_thresholds.get(float(threshold_percentage))
        
        if existing_alert:
            skip_info = _handle_existing_alert(existing_alert, threshold_percentage, threshold_amount)
            skipped_alerts.append(skip_info)
        else:
            alert_info = _create_new_alert(team_id, team, threshold_percentage, threshold_amount, db)
            created_alerts.append(alert_info)
    
    return created_alerts, skipped_alerts


def _build_response_data(team_id: UUID, team, total_cost_cap: float, 
                        created_alerts: list, skipped_alerts: list, deleted_alerts: list):
    """Build the response data structure."""
    return {
        "team_id": str(team_id),
        "team_name": team.name,
        "cost_cap": float(total_cost_cap),
        "created_count": len(created_alerts),
        "skipped_count": len(skipped_alerts),
        "deleted_count": len(deleted_alerts),
        "created_details": created_alerts,
        
        
        "skipped_details": skipped_alerts,
        "deleted_details": deleted_alerts
    }


def validate_thresholds(thresholds: List[float]) -> List[float]:
    if not thresholds:
        raise HTTPException(
            status_code=400,
            detail="At least one threshold percentage is required"
        )

    unique = list(set(thresholds))
    invalid = [p for p in unique if p <= 0 or p > 100]

    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid threshold percentages: {invalid}. Must be between 0 and 100"
        )

    return unique

def fetch_team_and_cap(db: Session, team_id: UUID):
    team = (
        db.query(Team)
        .filter(Team.id == team_id, Team.is_deleted == False)
        .first()
    )

    if not team:
        raise HTTPException(status_code=404, detail="Team not found")

    total_cap = db.query(
        func.coalesce(func.sum(TeamTokenQuotaPolicy.cost_cap), 0)
    ).filter(TeamTokenQuotaPolicy.team_id == team_id).scalar() or 0

    if total_cap == 0:
        raise HTTPException(
            status_code=400,
            detail="Team has no cost cap configured"
        )

    return team, total_cap


def process_recipients(db, team_id, recipients, user_email):
    created, skipped, deleted = [], [], []

    existing_rows = (
        db.query(BudgetAlertRecipient)
        .filter(BudgetAlertRecipient.team_id == team_id)
        .all()
    )

    existing_ids = {r.user_id for r in existing_rows}
    requested_ids = set(recipients)

    new_ids = requested_ids - existing_ids
    removed_ids = existing_ids - requested_ids
    kept_ids = existing_ids & requested_ids

    # Add new
    for uid in new_ids:
        db.add(BudgetAlertRecipient(
            team_id=team_id,
            user_id=uid,
            created_by=user_email
        ))
        created.append(str(uid))

    skipped.extend([str(uid) for uid in kept_ids])

    # Remove deleted
    if removed_ids:
        db.query(BudgetAlertRecipient).filter(
            BudgetAlertRecipient.team_id == team_id,
            BudgetAlertRecipient.user_id.in_(removed_ids)
        ).delete(synchronize_session=False)

        deleted.extend([str(uid) for uid in removed_ids])

    return {
        "created_count": len(created),
        "skipped_count": len(skipped),
        "deleted_count": len(deleted),
        "created_recipients": created,
        "skipped_recipients": skipped,
        "deleted_recipients": deleted
    }

def process_budget_alerts(db, team, team_id, unique_percentages, total_cost_cap):
    created, skipped, deleted = [], [], []

    existing_alerts = db.query(BudgetAlert).filter(
        BudgetAlert.team_id == team_id
    ).all()

    existing_map = {float(a.threshold_percentage): a for a in existing_alerts}
    requested_set = set(unique_percentages)

    # Delete alerts not in request
    for threshold, alert in list(existing_map.items()):
        if threshold not in requested_set:
            if alert.notified:
                skipped.append({
                    "threshold_percentage": threshold,
                    "threshold_amount": alert.threshold_amount,
                    "alert_name": alert.alert_name,
                    "reason": "already_notified — retained"
                })
            else:
                deleted.append({
                    "threshold_percentage": threshold,
                    "alert_name": alert.alert_name
                })
                db.delete(alert)
                existing_map.pop(threshold, None)

    # Create or skip alerts
    for percentage in unique_percentages:
        amount = (percentage / 100) * float(total_cost_cap)
        alert_name = f"{team.name}-budget-alert"

        existing = existing_map.get(float(percentage))
        if existing:
            reason = (
                "already_notified — will not recreate"
                if existing.notified
                else "already_exists"
            )
            skipped.append({
                "threshold_percentage": percentage,
                "threshold_amount": amount,
                "alert_name": existing.alert_name,
                "reason": reason
            })
        else:
            new_alert = BudgetAlert(
                team_id=team_id,
                organization_id=team.organization_id,
                alert_name=alert_name,
                threshold_percentage=percentage,
                threshold_amount=amount,
                cost_consumed=0,
                notified=False,
                created_by="system",
                updated_by="system"
            )
            db.add(new_alert)

            created.append({
                "threshold_percentage": percentage,
                "threshold_amount": amount,
                "alert_name": alert_name
            })

    return {
        "created_count": len(created),
        "skipped_count": len(skipped),
        "deleted_count": len(deleted),
        "created_details": created,
        "skipped_details": skipped,
        "deleted_details": deleted
    }

@team_router.post("/budget-alerts/{team_id}", response_model=None)
def create_budget_alert(
    team_id: UUID,
    request: Request,
    threshold_percentages: List[float] = Query(...),
    recipients: List[UUID] = Query(...),
    db: Session = Depends(get_db)
):
    user_email = getattr(request.state, "user_email", "system")

    try:
        # 1. Validate inputs
        unique_percentages = validate_thresholds(threshold_percentages)

        # 2. Fetch team + cost cap
        team, total_cost_cap = fetch_team_and_cap(db, team_id)

        # 3. Handle recipients
        recipient_summary = process_recipients(
            db=db,
            team_id=team_id,
            recipients=recipients,
            user_email=user_email
        )

        # 4. Handle budget alerts
        alert_summary = process_budget_alerts(
            db=db,
            team=team,
            team_id=team_id,
            unique_percentages=unique_percentages,
            total_cost_cap=total_cost_cap
        )

        db.commit()

        return success_response(
            data={
                "team_id": str(team_id),
                "team_name": team.name,
                "cost_cap": float(total_cost_cap),
                **alert_summary,
                "recipient_summary": recipient_summary
            },
            message="Budget alerts processed successfully.",
            status_code=201
        )

    except Exception as e:
        db.rollback()
        return error_response(
            message="Failed to process budget alerts",
            error_key="INTERNAL_ERROR",
            details=str(e),
            status_code=500
        )

@team_router.get("/budget-alerts/{team_id}", response_model=None,dependencies=[Depends(role_guard(["SUPER_ADMIN","ADMIN","finops_admin"]))])
def get_budget_alerts(
    team_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get all budget alerts for a specific team with current consumption details and recipients.
    """
    try:
        # Verify team exists
        team = db.query(Team).filter(Team.id == team_id, Team.is_deleted == False).first()
        if not team:
            return error_response(
                message="Team not found",
                error_key="TEAM_NOT_FOUND",
                status_code=404
            )

        # Get team's cost cap
        cost_cap = db.query(func.coalesce(func.sum(TeamTokenQuotaPolicy.cost_cap), 0)) \
            .filter(TeamTokenQuotaPolicy.team_id == team_id).scalar() or 0

        # Get current cost consumed
        cost_consumed = db.query(func.coalesce(func.sum(TokenQuotaCounter.cost_consumed), 0)) \
            .filter(TokenQuotaCounter.team_id == team_id).scalar() or 0

        quota_policy = db.query(TeamTokenQuotaPolicy).filter(
            TeamTokenQuotaPolicy.team_id == team_id
        ).order_by(TeamTokenQuotaPolicy.effective_from.desc()).first()

        # Fetch all alerts for this team
        alerts = db.query(BudgetAlert).filter(
            BudgetAlert.team_id == team_id
        ).order_by(BudgetAlert.threshold_percentage.asc()).all()

        # ===================== Recipients Fetching =====================
        recipients_data = []
        recipients = (
            db.query(BudgetAlertRecipient)
            .filter(BudgetAlertRecipient.team_id == team_id)
            .all()
        )

        if recipients:
            user_ids = [r.user_id for r in recipients]
            users = db.query(User).filter(User.id.in_(user_ids)).all()
            user_map = {u.id: u for u in users}

            for r in recipients:
                user = user_map.get(r.user_id)
                recipients_data.append({
                    "id": str(r.id),
                    "user_id": str(r.user_id),
                    "user_email": user.email if user else None,
                    "user_name": f"{user.first_name} {user.last_name}".strip() if user else None,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "updated_at": r.updated_at.isoformat() if r.updated_at else None,
                    "created_by": r.created_by,
                    "updated_by": r.updated_by
                })

        if not alerts:
            return success_response(
                data={
                    "team_id": str(team_id),
                    "team_name": team.name,
                    "cost_cap": float(cost_cap),
                    "cost_consumed": float(cost_consumed),
                    "auto_renew": quota_policy.auto_renew if quota_policy else None,
                    "effective_to": quota_policy.effective_to.isoformat() if quota_policy and quota_policy.effective_to else None,
                    "alerts": [],
                    "recipients": recipients_data  
                },
                message="No budget alerts found for this team"
            )

        # Format alerts response
        alerts_data = []
        for alert in alerts:
            alerts_data.append({
                "id": alert.id,
                "alert_name": alert.alert_name,
                "threshold_percentage": float(alert.threshold_percentage),
                "threshold_amount": float(alert.threshold_amount),
                "cost_consumed": float(alert.cost_consumed),
                "currency": alert.currency,
                "status": alert.status,
                "notified": alert.notified,
                "alert_type": alert.alert_type,
                "created_at": alert.created_at.isoformat() if alert.created_at else None,
                "updated_at": alert.updated_at.isoformat() if alert.updated_at else None
            })

        return success_response(
            data={
                "team_id": str(team_id),
                "team_name": team.name,
                "cost_cap": float(cost_cap),
                "cost_consumed": float(cost_consumed),
                "auto_renew": quota_policy.auto_renew if quota_policy else None,
                "effective_to": quota_policy.effective_to.isoformat() if quota_policy and quota_policy.effective_to else None,
                "consumption_percentage": round((float(cost_consumed) / float(cost_cap) * 100), 2) if cost_cap > 0 else 0,
                "alerts": alerts_data,
                "recipients": recipients_data   
            },
            message="Budget alerts fetched successfully"
        )

    except Exception as e:
        return error_response(
            message="Failed to fetch budget alerts",
            error_key="INTERNAL_ERROR",
            details=str(e),
            status_code=500
        )
 

@team_router.post("/llm-config", response_model=None, dependencies=[Depends(role_guard(["SUPER_ADMIN","ADMIN","finops_admin"]))])
def create_update_delete_team_llm_config(
    body: TeamsLLMConfigRequest,
    request: Request,
    db: Session = Depends(get_db)
):
    created_configs, skipped_configs, deleted_configs = [], [], []
    user_email = getattr(request.state, "user_email", "system")

    try:
        model_ids = [str(cfg.model_id) for cfg in body.configs]
        duplicates = [mid for mid in set(model_ids) if model_ids.count(mid) > 1]

        if duplicates:
            return error_response(
                message="Duplicate model_id(s) found in request",
                error_key="VALIDATION_ERROR",
                details={"duplicate_model_ids": duplicates},
                status_code=400
            )
        
        existing_configs = db.query(TeamsLLMConfig).filter_by(team_id=body.team_id).all()
        existing_map = {str(cfg.model_id): cfg for cfg in existing_configs}
        incoming_model_ids = {str(cfg.model_id) for cfg in body.configs}

        # DELETE configs not in request
        for model_id, existing in existing_map.items():
            if model_id not in incoming_model_ids:
                deleted_configs.append({"model_id": model_id})
                db.delete(existing)

        # CREATE or UPDATE configs
        for cfg in body.configs:
            key_managed_by = getattr(cfg, "key_managed_by", None)

            existing = existing_map.get(str(cfg.model_id))

            # -------------------------------
            # STEP 1: Load old config if exists
            # -------------------------------
            if existing:
                try:
                    old_config = json.loads(decrypt_text(existing.config))
                except Exception:
                    old_config = {}
            else:
                old_config = {}

            # -------------------------------
            # STEP 2: Merge new config with old
            # -------------------------------
            merged_config = {**old_config, **cfg.config}

            # -------------------------------
            # STEP 3: API KEY LOGIC
            # -------------------------------
            if key_managed_by == "YASH-Managed":
                # always override with managed key
                merged_config["api_key"] = settings.YASH_MANAGED_API_KEY
            
            elif "api_key" in cfg.config:
                # user-provided API key
                merged_config["api_key"] = cfg.config["api_key"]
            
            # Else → Do NOT change old API key

            # -------------------------------
            # STEP 4: Save merged config
            # -------------------------------
            encrypted_config = encrypt_text(json.dumps(merged_config))

            if existing:
                # CHECK if anything changed
                if (
                    existing.config == encrypted_config
                    and existing.is_active == cfg.is_active
                    and existing.key_managed_by == key_managed_by
                ):
                    skipped_configs.append({
                        "model_id": str(cfg.model_id),
                        "reason": "already_exists_with_same_config"
                    })
                else:
                    existing.config = encrypted_config
                    existing.is_active = cfg.is_active
                    existing.key_managed_by = key_managed_by
                    existing.updated_by = user_email

                    skipped_configs.append({
                        "model_id": str(cfg.model_id),
                        "reason": "updated_existing_config"
                    })

            else:
                new_config = TeamsLLMConfig(
                    team_id=body.team_id,
                    model_id=cfg.model_id,
                    config=encrypted_config,
                    is_active=cfg.is_active,
                    key_managed_by=key_managed_by,
                    created_by=user_email,
                    updated_by=user_email,
                )
                db.add(new_config)
                created_configs.append({"model_id": str(cfg.model_id)})

        db.commit()

        return success_response(
            data={
                "team_id": str(body.team_id),
                "created_count": len(created_configs),
                "skipped_count": len(skipped_configs),
                "deleted_count": len(deleted_configs),
                "created_details": created_configs,
                "skipped_details": skipped_configs,
                "deleted_details": deleted_configs
            },
            message="Team LLM configs processed successfully.",
            status_code=201
        )

    except Exception as e:
        db.rollback()
        return error_response(
            message="Failed to process team LLM configs",
            error_key="INTERNAL_ERROR",
            details=str(e),
            status_code=500
        )
    
@team_router.get("/team/{team_id}/model/{model_id}/api-key", response_model=dict, dependencies=[Depends(role_guard(["SUPER_ADMIN","ADMIN","finops_admin"]))])
def get_team_model_api_key(team_id: UUID, model_id: UUID, db: Session = Depends(get_db)):
    try:
        # Prefer selecting TeamsLLMConfig directly (easier to reason about)
        config_entry = db.query(TeamsLLMConfig).filter(
            TeamsLLMConfig.team_id == team_id,
            TeamsLLMConfig.model_id == model_id
        ).first()
 
        if not config_entry:
            return error_response(
                message="LLM config not found for provided team_id and model_id",
                error_key="NOT_FOUND",
                status_code=404,
            )
 
        raw = config_entry.config
        if raw is None or (isinstance(raw, str) and raw.strip() == ""):
            return error_response(
                message="Config column is empty or null for this team/model",
                error_key="MISSING_CONFIG",
                status_code=404,
            )
 
        # Accept str or bytes (some ORMs return bytes for bytea)
        if isinstance(raw, bytes):
            try:
                raw = raw.decode()  # try decode
            except Exception:
                # if it really is binary, consider base64 or fix storage
                return error_response(
                    message="Config appears to be binary/unexpected format",
                    error_key="INVALID_CONFIG_FORMAT",
                    status_code=500,
                )
 
        # Decrypt
        try:
            decrypted = decrypt_text(raw)
        except Exception as e:
            # log exception server-side
            return error_response(
                message="Failed to decrypt stored config",
                error_key="DECRYPTION_ERROR",
                details=str(e),
                status_code=500,
            )
 
        # Parse JSON if possible
        try:
            cfg = json.loads(decrypted)
        except Exception:
            cfg = {"raw_config": decrypted}
 
        api_key = cfg.get("api_key") if isinstance(cfg, dict) else None
        if not api_key:
            return error_response(
                message="API key not present in decrypted config",
                error_key="API_KEY_MISSING",
                status_code=404,
            )
 
        return success_response(data={"team_id": str(team_id), "model_id": str(model_id), "api_key": api_key},
                                message="API key fetched successfully")
 
    except Exception as e:
        # logger.exception("Unexpected error while fetching API key")
        return error_response(
            message="Failed to fetch API key",
            error_key="DB_ERROR",
            details=str(e),
            status_code=500
        )
    
def first_day_of_month(dt: datetime = None) -> datetime:
    if dt is None:
        dt = datetime.now(IST)
    if dt.tzinfo is None:
        dt = IST.localize(dt)
    else:
        dt = dt.astimezone(IST)
    return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
async def create_next_month_quota_policy(db: Session, team_id: UUID, current_policy: TeamTokenQuotaPolicy):
    """Create next month's quota policy + counter for a team"""
    now = datetime.now(IST)

    # Calculate next month
    next_month = now + relativedelta(months=1)
    effective_from = first_day_of_month(next_month)
    effective_to = last_day_of_month(next_month)

    # Create new policy (copy values from current one)
    new_policy = TeamTokenQuotaPolicy(
        team_id=team_id,
        token_cap=current_policy.token_cap,
        cost_cap=current_policy.cost_cap,
        auto_renew=True,
        effective_from=effective_from.replace(tzinfo=None),
        effective_to=effective_to.replace(tzinfo=None),
        created_by="system_monthly_rollover",
        updated_by="system_monthly_rollover"
    )
    db.add(new_policy)
    db.flush()  # to get new_policy.id

    # Create fresh counter for the new period
    new_counter = TokenQuotaCounter(
        team_id=team_id,
        team_token_quota_policy_id=new_policy.id,
        tokens_consumed=0,
        cost_consumed=0.0,
        prompt_token_cost=0.0,
        completion_token_cost=0.0,
        thought_token_cost=0.0,
        created_by="system_monthly_rollover"
    )
    db.add(new_counter)
    db.flush()

    await set_team_quota(                     
        team_id=str(team_id),
        cap=new_policy.cost_cap or Decimal("0"),
        consumed=Decimal("0")                 
    )
    print(f"Redis quota reset for team {team_id}: cap={new_policy.cost_cap}")

    return new_policy, new_counter


@team_router.post("/monthly-rollover", dependencies=[Depends(role_guard(["SUPER_ADMIN"]))])
async def trigger_monthly_quota_rollover(db: Session = Depends(get_db)):
    """
    Simple endpoint to trigger monthly rollover for all auto_renew = True teams
    Run this once per month (e.g., on 1st of every month via cron)
    """
    try:
        now = datetime.now(IST)
        today = now.date()

        # Only run on the 1st or 2nd of the month to avoid duplicates
        if today.day not in (1, 2):
            return {
                "detail": f"Rollover only allowed on 1st or 2nd of month. Today is {today.day}"
            }

        # Get last moment of CURRENT month (as datetime, not date)
        current_month_last_day = last_day_of_month(now).replace(tzinfo=None)

        active_auto_renew_policies = db.query(TeamTokenQuotaPolicy).filter(
            TeamTokenQuotaPolicy.auto_renew == True,
            TeamTokenQuotaPolicy.effective_to <= current_month_last_day  
        ).all()

        created_count = 0
        for policy in active_auto_renew_policies:
            team_id = policy.team_id

            next_month_start = first_day_of_month(now + relativedelta(months=1)).replace(tzinfo=None)

            # Avoid duplicates: check if policy for next month already exists
            already_exists = db.query(TeamTokenQuotaPolicy).filter(
                TeamTokenQuotaPolicy.team_id == team_id,
                TeamTokenQuotaPolicy.effective_from == next_month_start
            ).first()

            if not already_exists:
                new_policy, new_counter = await create_next_month_quota_policy(db, team_id, policy)
                created_count += 1

                # Optional: commit per team or batch
                db.commit()
                print(f"Created new quota for team {team_id}: {new_policy.id}")

        db.commit()  # final commit if batching

        return {
            "message": "Monthly quota rollover completed",
            "teams_processed": len(active_auto_renew_policies),
            "new_policies_created": created_count,
            "triggered_at": now.isoformat()
        }

    except Exception as e:
        db.rollback()
        raise e