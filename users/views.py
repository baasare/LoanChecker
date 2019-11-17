from django.contrib import messages
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import AuthenticationForm
from django.shortcuts import render, redirect

from .forms import SignUpForm


# Create your views here.


def index(request):
    return render(request=request,
                  template_name="users/index.html",
                  )


def signup(request):
    if request.method == "POST":
        form = SignUpForm(data=request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=raw_password)
            login(request, user)
            return redirect('index')
        else:
            for field in form:
                for error in field.errors:
                    # messages.error(request, error)
                    print("Field: ", end='')
                    print(field)
                    print()
                    print("Error:", end='')
                    print(error)

            args = {'form': form}
            return render(request, 'users/signup.html', args)
    else:
        form = SignUpForm()

    args = {'form': form}
    return render(request, 'users/signup.html', args)


def signin(request):
    if request.method == 'POST':
        form = AuthenticationForm(request=request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                if form.cleaned_data.get('remember_me'):
                    request.session.set_expiry(1209600)  # 2 weeks
                login(request, user)
                messages.success(request, "You have successfully logged in")
                return redirect('index')
            else:
                messages.error(request, "Invalid username or password.")
        else:
            messages.error(request, "Invalid username or password.")
    else:
        form = AuthenticationForm()

    return render(request=request,
                  template_name="users/signin.html",
                  context={"form": form})


def signout(request):
    logout(request)
    messages.info(request, "Logged out successfully!")
    return redirect("index")
